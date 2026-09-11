// sm120_mma_tma_transform.hpp
//
// Producer-warp ("transform-warp") variant of the sm120 bf16 x NVFP4 mixed-input mainloop.
//
// v3 (sm120_mma_tma_mixed_input.hpp) dequantises on the 8 MMA warps, between ldmatrix and mma.sync,
// and every B fragment is converted 4x (once per M-warp). Here the conversion moves to the otherwise
// idle warps of the producer warpgroup (warps 2..3, or 1..3 with a static tile scheduler):
//
//   TMA warp  --(PipelineTmaAsync, Stages)-->  [A bf16 | Bp packed nibbles | S bf16 scales] per stage
//   transform warps: wait stage -> read Bp/S -> e2m1 LUT -> bf16 * scale -> STS.128 into a bf16 B tile
//                    in a separate ring (PipelineAsync, RingStages), release the stage
//   MMA warps: the UNMODIFIED dense sm120 loop (ldmatrix A from the stage, ldmatrix B from the ring,
//              mma.sync), then release the stage (A) and the ring entry (B).
//
// Shared memory (128x128x64 tile, bf16 scales): stage = 16384 (A) + 4096 (Bp) + 1024 (S) = 21504 B,
// ring entry = 16384 B. 2 stages + 2 entries = 75776 B of the 87040 B left after the epilogue's
// 14336 B. 3 stages + 2 entries (97280 B) does not fit, and the ring (not the stage count) bounds the
// transform's lead over the MMA warps at one k-tile either way.
//
// Packed layouts (host side; no repack of the weight at all):
//   Bp[N][K/8] uint32: MLX / ModelOpt native: element k at bits 4*(k%8) .. +3 of word k/8.
//   Sp[N/2][K/8] uint16 (bf16 bits of the e4m3 scale):
//       Sp[p][8*kt + (n%2)*4 + gg] = scale(n = 2p + n%2, group g = 4*kt + gg),  g = k/16.
//     i.e. per row pair and k-tile the 8 scales are one 16 B TMA row (8 B would be below the TMA minimum).
//
// Knobs (-D):
//   NVFP4_TW_MODE  2 = real (default);
//                  1 = transform warps skip the conversion (wrong numerics; timing probe: same loads, stores,
//                      barriers and ring dependency);
//                  0 = full conversion but NO ring dependency: the MMA warps never wait for the ring and the
//                      transform warps never wait for free entries (wrong numerics; measures the pure
//                      issue-slot / smem interference of the conversion ALU on the MMA warps);
//                  3 = neither conversion nor dependency (wrong numerics; the "idle transform warps" floor).
//   NVFP4_TW_RING_K16  1 = hand the ring over per k16 quarter: each 16 KB ring entry keeps 4 full/empty
//                      barrier pairs, the MMA warps wait per k_block and release a quarter as soon as its
//                      ldmatrix is in registers, and the transform warps arrive after each quarter. Same
//                      shared memory; only the exposed transform latency on the MMA critical path shrinks
//                      from a whole tile's conversion to a quarter's.
//   NVFP4_TW_CHECK: trap if the hand-computed swizzled store address disagrees with CuTe's layout.

#pragma once

#ifndef NVFP4_TW_MODE
#define NVFP4_TW_MODE 2
#endif
#ifndef NVFP4_TW_RING_K16
#define NVFP4_TW_RING_K16 0
#endif
#define NVFP4_TW_DEP  (NVFP4_TW_MODE == 1 || NVFP4_TW_MODE == 2)
#define NVFP4_TW_CONV (NVFP4_TW_MODE == 0 || NVFP4_TW_MODE == 2)

#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/trace.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

namespace cutlass::gemm {

template <int Stages_, int RingStages_, int TransformWarps_, int SchedulerPipelineStageCount_, class ClusterShape_, class KernelSchedule_>
struct MainloopSm120TmaWarpSpecializedTransform {
  constexpr static int Stages = Stages_;
  constexpr static int RingStages = RingStages_;
  constexpr static int TransformWarps = TransformWarps_;
  constexpr static int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule_;
  constexpr static int PipelineAsyncMmaStages = 0;
  using ArchTag = arch::Sm120;
};

// Kernel schedule tag for the transform-warp kernel layer (sm120_gemm_tma_ws_cooperative_transform.hpp).
// Deliberately NOT derived from KernelTmaWarpSpecializedCooperative so that CUTLASS's sm90 kernel
// specialisation does not also match.
struct KernelTmaWarpSpecializedCooperativeTransformSm120Base {};
template <int SchedulerPipelineStageCount_ = 2>
struct KernelTmaWarpSpecializedCooperativeTransformSm120 : KernelTmaWarpSpecializedCooperativeTransformSm120Base {
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
};

} // namespace cutlass::gemm

namespace cutlass::gemm::collective {
using namespace cute;

namespace nvfp4_tw_detail {

CUTLASS_DEVICE __nv_bfloat162 as_bf162(uint32_t u) { __nv_bfloat162 r; *reinterpret_cast<uint32_t*>(&r) = u; return r; }
CUTLASS_DEVICE uint32_t as_u32(__nv_bfloat162 v) { return *reinterpret_cast<uint32_t*>(&v); }
CUTLASS_DEVICE uint32_t dup_lo16(uint32_t x) { return __byte_perm(x, x, 0x1010); }
CUTLASS_DEVICE uint32_t dup_hi16(uint32_t x) { return __byte_perm(x, x, 0x3232); }

// 8 e2m1 nibbles (element i at bits 4i..4i+3) -> 4 bf16x2 words in element order, times a bf16 scale
// (exact: e2m1 x e4m3 needs <= 6 significant bits). CUTLASS's prmt-LUT converter: 13 instructions per 4 values.
CUTLASS_DEVICE void dequant8(uint32_t w, __nv_bfloat162 scale, uint32_t& o0, uint32_t& o1, uint32_t& o2, uint32_t& o3) {
#if !NVFP4_TW_CONV
  // timing probe: no conversion, wrong numerics
  o0 = w; o1 = w ^ as_u32(scale); o2 = w + 1; o3 = w ^ 0x5555u;
#else
  using Cvt = cutlass::NumericArrayConverter<cutlass::bfloat16_t, cutlass::float_e2m1_t, 8>;
  cutlass::Array<cutlass::float_e2m1_t, 8> src;
  *reinterpret_cast<uint32_t*>(&src) = w;
  cutlass::Array<cutlass::bfloat16_t, 8> v = Cvt::convert(src);
  uint32_t const* vv = reinterpret_cast<uint32_t const*>(&v);
  o0 = as_u32(__hmul2(as_bf162(vv[0]), scale));
  o1 = as_u32(__hmul2(as_bf162(vv[1]), scale));
  o2 = as_u32(__hmul2(as_bf162(vv[2]), scale));
  o3 = as_u32(__hmul2(as_bf162(vv[3]), scale));
#endif
}

} // namespace nvfp4_tw_detail

template <
  int Stages,
  int RingStages,
  int TransformWarps,
  int SchedulerPipelineStageCount,
  class ClusterShape,
  class KernelScheduleType,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_>
struct CollectiveMma<
    MainloopSm120TmaWarpSpecializedTransform<Stages, RingStages, TransformWarps, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>,
    TileShape_, ElementA_, StrideA_, ElementB_, StrideB_, TiledMma_,
    GmemTiledCopyA_, SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_,
    GmemTiledCopyB_, SmemLayoutAtomB_, SmemCopyAtomB_, TransformB_> {

  using DispatchPolicy = MainloopSm120TmaWarpSpecializedTransform<Stages, RingStages, TransformWarps, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>;
  using TileShape = TileShape_;
  using ElementA  = ElementA_;                 // bf16
  using StrideA   = StrideA_;
  using ElementB  = ElementB_;                 // uint32_t: 8 packed e2m1 codes, MLX-native order
  using StrideB   = StrideB_;                  // kept for the kernel layer; TMA descriptors are built from (N, K/8) / (N/2, K/8)
  using ElementScale = uint16_t;               // bf16 bits
  using TiledMma  = TiledMma_;
  using ElementBMma = typename TiledMma::ValTypeB;   // bf16
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;    // bf16 ring atom (the dense kernel's B atom)
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using RuntimeDataTypeA = void*;
  using RuntimeDataTypeB = void*;

  static constexpr int ThreadCount = size(TiledMma{});
  static constexpr int NumTransformWarps = TransformWarps;
  static constexpr int NumTransformThreads = NumTransformWarps * NumThreadsPerWarp;
  static_assert(NumTransformWarps >= 1 && NumTransformWarps <= 3, "transform warps live in the producer warpgroup (warps 1..3)");

  static constexpr int BLK_M = size<0>(TileShape{});
  static constexpr int BLK_N = size<1>(TileShape{});
  static constexpr int BLK_K = size<2>(TileShape{});
  static_assert(BLK_K == 64, "prototype: K tile of 64 (16 B of scales per row pair per k-tile is the TMA minimum box row)");
  static_assert(BLK_N % 16 == 0, "row pairs and 8-row swizzle atoms");
  static constexpr int WPR   = BLK_K / 8;      // packed words per row per k-tile (8)
  static constexpr int SPR   = BLK_K / 16;     // scales per row per k-tile (4)
  static constexpr int SROW2 = 2 * SPR;        // scale elements per row pair per k-tile (8 = 16 B)
  static constexpr int ITEMS = BLK_N * 2;      // transform work items per k-tile: (row, k32 half) = 4 words = 32 values
  static constexpr int KBLOCKS = BLK_K / 16;   // k16 steps per k-tile (4)
  // ring hand-over granularity: 1 = whole 16 KB entry, KBLOCKS = one barrier pair per k16 quarter of the entry
  static constexpr int RingSub = NVFP4_TW_RING_K16 ? KBLOCKS : 1;
  static constexpr bool kDep = NVFP4_TW_DEP;

  using MainloopPipeline = cutlass::PipelineTmaAsync<Stages>;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState  = cutlass::PipelineState<Stages>;
  using RingPipeline = cutlass::PipelineAsync<RingStages * RingSub>;
  using RingPipelineState = typename RingPipeline::PipelineState;

  static constexpr int NumProducerThreadEvents = 1;

  static_assert(cute::is_same_v<ElementB, uint32_t>, "ElementB must be the packed uint32 word type");
  static_assert(cute::is_same_v<ElementBMma, cutlass::bfloat16_t>, "MMA B type must be bf16");
  static_assert(cute::is_same_v<ElementA, cutlass::bfloat16_t>, "A must be bf16");
  static_assert(decltype(size<2>(typename TiledMma::AtomShape_MNK{}))::value == 16, "expects the m16n8k16 bf16 atom");

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(not cute::is_void_v<SmemCopyAtomA>, "SM120 mainloop must specify a copy atom for A operand smem->rmem reads.");
  static_assert(not cute::is_void_v<SmemCopyAtomB>, "SM120 mainloop must specify a copy atom for B operand smem->rmem reads.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  // bf16 B ring: exactly the dense mainloop's SmemLayoutB with RingStages instead of Stages (K-major, SW128).
  using SmemLayoutBB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<RingStages>{}),
      Step<_1,_2,_3>{}));
  // Packed words (BLK_N, WPR, PIPE) row-major: one 32 B row per n.
  using SmemLayoutBp = Layout<Shape<Int<BLK_N>, Int<WPR>, Int<Stages>>, Stride<Int<WPR>, _1, Int<BLK_N * WPR>>>;
  // Scales (BLK_N/2, SROW2, PIPE) row-major: one 16 B row per row pair.
  using SmemLayoutS = Layout<Shape<Int<BLK_N / 2>, Int<SROW2>, Int<Stages>>, Stride<Int<SROW2>, _1, Int<(BLK_N / 2) * SROW2>>>;

  static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
  static_assert(Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(RingStages >= 2, "the bf16 B ring needs at least 2 entries");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD>, "GmemTiledCopyA - only SM90_TMA_LOAD (no multicast) in this prototype.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD>, "GmemTiledCopyB - only SM90_TMA_LOAD (no multicast) in this prototype.");
  static_assert(cute::size(ClusterShape{}) == 1, "no clusters in this prototype");

  using TmaInternalElementA = uint_bit_t<sizeof_bits_v<ElementA>>;

  static constexpr uint32_t TmaTransactionBytesMK = static_cast<uint32_t>(
      cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutA{})) * sizeof_bits<ElementA>::value));
  static constexpr uint32_t TmaTransactionBytesNK = static_cast<uint32_t>(
      BLK_N * WPR * sizeof(uint32_t) + (BLK_N / 2) * SROW2 * sizeof(ElementScale));
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

  static constexpr int StageBytes = static_cast<int>(TmaTransactionBytes);
  static constexpr int RingEntryBytes = BLK_N * BLK_K * static_cast<int>(sizeof(ElementBMma));

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      alignas(1024) cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(1024) cute::array_aligned<ElementBMma, cute::cosize_v<SmemLayoutBB>> smem_BB;
      alignas(128)  cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutBp>> smem_Bp;
      alignas(128)  cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutS>> smem_S;
    } tensors;
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      alignas(16) typename MainloopPipeline::SharedStorage tma;
      alignas(16) typename RingPipeline::SharedStorage ring;
    } pipeline_storage;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  struct Arguments {
    ElementA const* ptr_A{nullptr};
    StrideA dA{};
    uint32_t const* ptr_B{nullptr};       // Bp[N][K/8] (MLX native)
    ElementScale const* ptr_S{nullptr};   // Sp[N/2][K/8]
  };

  using StrideBp = Stride<int64_t, Int<1>, int64_t>;
  using StrideSp = Stride<int64_t, Int<1>, int64_t>;

  struct Params {
    using TMA_A = decltype(make_tma_copy(
        GmemTiledCopyA{},
        make_tensor(recast_ptr<TmaInternalElementA>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,0),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        _1{}));
    using TMA_B = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(static_cast<uint32_t const*>(nullptr), make_shape(int32_t(0), int32_t(0), int32_t(0)), StrideBp{}),
        SmemLayoutBp{}(_,_,0),
        make_shape(Int<BLK_N>{}, Int<WPR>{}),
        _1{}));
    using TMA_S = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(static_cast<ElementScale const*>(nullptr), make_shape(int32_t(0), int32_t(0), int32_t(0)), StrideSp{}),
        SmemLayoutS{}(_,_,0),
        make_shape(Int<BLK_N / 2>{}, Int<SROW2>{}),
        _1{}));
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_S tma_load_s;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    (void) workspace;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    auto ptr_A = recast_ptr<TmaInternalElementA>(args.ptr_A);
    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    typename Params::TMA_A tma_load_a = make_tma_copy(
        GmemTiledCopyA{}, tensor_a, SmemLayoutA{}(_,_,cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})), _1{});

    int32_t const k8 = K / 8;
    Tensor tensor_b = make_tensor(args.ptr_B, make_layout(make_shape(N, k8, L),
        make_stride(int64_t(k8), Int<1>{}, int64_t(N) * int64_t(k8))));
    typename Params::TMA_B tma_load_b = make_tma_copy(
        SM90_TMA_LOAD{}, tensor_b, SmemLayoutBp{}(_,_,cute::Int<0>{}),
        make_shape(Int<BLK_N>{}, Int<WPR>{}), _1{});

    int32_t const n2 = N / 2;
    Tensor tensor_s = make_tensor(args.ptr_S, make_layout(make_shape(n2, k8, L),
        make_stride(int64_t(k8), Int<1>{}, int64_t(n2) * int64_t(k8))));
    typename Params::TMA_S tma_load_s = make_tma_copy(
        SM90_TMA_LOAD{}, tensor_s, SmemLayoutS{}(_,_,cute::Int<0>{}),
        make_shape(Int<BLK_N / 2>{}, Int<SROW2>{}), _1{});

    return { tma_load_a, tma_load_b, tma_load_s, TmaTransactionBytes, TmaTransactionBytesMK, TmaTransactionBytesNK };
  }

  template<class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, [[maybe_unused]] Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_MNKL;
    bool implementable = true;
    implementable = implementable && (N % BLK_N == 0) && (K % BLK_K == 0) && (L == 1);
    constexpr int min_tma_aligned_elements_A = 128 / cutlass::sizeof_bits<ElementA>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: transform-warp prototype needs N % TileN == 0, K % TileK == 0, L == 1, 16B-aligned A.\n");
    }
    return implementable;
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_s.get_tma_descriptor());
  }

  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    auto [M, N, K, L] = problem_shape_MNKL;
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,L));                 // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K/8,L));               // (n,k8,l) words
    Tensor mS_nkl = mainloop_params.tma_load_s.get_tma_tensor(make_shape(N/2,K/8,L));             // (n2,k8,l) scales
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});         // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, make_shape(Int<BLK_N>{}, Int<WPR>{}), make_coord(_,_,_));      // (BLK_N,WPR,n,k,l)
    Tensor gS_nkl = local_tile(mS_nkl, make_shape(Int<BLK_N / 2>{}, Int<SROW2>{}), make_coord(_,_,_)); // (BLK_N/2,SROW2,n,k,l)
    return cute::make_tuple(gA_mkl, gB_nkl, gS_nkl);
  }

  // ---- TMA producer warp (one elected thread) ----
  template <class TensorA, class TensorB, class TensorS, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB, TensorS> const& load_inputs,
      BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors) {
    (void) thread_idx; (void) block_rank_in_cluster;
    int lane_predicate = cute::elect_one_sync();
    if (lane_predicate) {
      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_Bp.data()), SmemLayoutBp{});      // (BLK_N,WPR,PIPE)
      Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_S.data()), SmemLayoutS{});        // (BLK_N/2,SROW2,PIPE)

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gB_nkl = get<1>(load_inputs);
      Tensor gS_nkl = get<2>(load_inputs);

      auto block_tma_a = mainloop_params.tma_load_a.get_slice(0);
      auto block_tma_b = mainloop_params.tma_load_b.get_slice(0);
      auto block_tma_s = mainloop_params.tma_load_s.get_slice(0);

      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                  // (BLK_M,BLK_K,k)
      Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                  // (BLK_N,WPR,k)
      Tensor gS = gS_nkl(_,_,n_coord,_,l_coord);                                                  // (BLK_N/2,SROW2,k)

      Tensor tAgA = block_tma_a.partition_S(gA);
      Tensor tAsA = block_tma_a.partition_D(sA);
      Tensor tBgB = block_tma_b.partition_S(gB);
      Tensor tBsB = block_tma_b.partition_D(sB);
      Tensor tSgS = block_tma_s.partition_S(gS);
      Tensor tSsS = block_tma_s.partition_D(sS);

      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        pipeline.producer_acquire(smem_pipe_write);
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
        int write_stage = smem_pipe_write.index();
        copy(mainloop_params.tma_load_a.with(*tma_barrier, 0), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(mainloop_params.tma_load_b.with(*tma_barrier, 0), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        copy(mainloop_params.tma_load_s.with(*tma_barrier, 0), tSgS(_,_,_,*k_tile_iter), tSsS(_,_,_,write_stage));
        ++k_tile_iter;
        ++smem_pipe_write;
      }
    }
  }

  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();
    if (lane_predicate) {
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  // ---- transform warps: packed nibbles + scales (TMA stage) -> bf16 B tile (ring entry) ----
  // Work item = (row n, k32 half h): one LDS.128 of 4 packed words (32 consecutive k), one LDS.32 of the two
  // bf16 scales for those 32 k, 4 x (LUT convert + 4 HMUL2), 4 STS.128 into the SW128 K-major bf16 tile.
  // Bank conflicts: the LDS addresses are 16 B / 4 B contiguous across lanes; the STS.128 addresses of the
  // 8 lanes in a phase land in 8 distinct 16 B chunks because rows 0..3 XOR chunk bits {0..3} with h in bit 2.
  // With NVFP4_TW_RING_K16 the lanes take consecutive rows at the same half instead (2-way LDS.128 conflicts,
  // 4 loads per tile) so that a quarter of the tile is complete after a quarter of the work.
  CUTLASS_DEVICE void
  transform(
      MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      RingPipeline ring,
      RingPipelineState ring_write,
      int k_tile_count,
      int transform_thread_idx,
      TensorStorage& shared_tensors) {
    using namespace nvfp4_tw_detail;

    uint8_t* bb_base = reinterpret_cast<uint8_t*>(shared_tensors.smem_BB.data());
#ifdef NVFP4_TW_CHECK
    Tensor sBB = make_tensor(make_smem_ptr(shared_tensors.smem_BB.data()), SmemLayoutBB{});     // (BLK_N,BLK_K,RING)
    Tensor sBB_pi = as_position_independent_swizzle_tensor(sBB);
#endif

    // convert packed word `w` (8 consecutive k starting at k0 of row n) into ring entry `bb`
    auto store8 = [&](uint8_t* bb, int n, int k0, uint32_t w, __nv_bfloat162 scale, [[maybe_unused]] int we) {
      uint32_t o0, o1, o2, o3;
      dequant8(w, scale, o0, o1, o2, o3);
      // byte offset of (n, k0) in the K-major tile; Swizzle<3,4,3>: 16 B chunk index (bits 4..6) ^= row % 8
      uint32_t const off = (static_cast<uint32_t>(n) * (BLK_K * 2) + k0 * 2) ^ (static_cast<uint32_t>(n & 7) << 4);
#ifdef NVFP4_TW_CHECK
      {
        ElementBMma* want = &sBB_pi(n, k0, we);
        if (reinterpret_cast<uint8_t*>(want) != bb + off) { __trap(); }
      }
#endif
      *reinterpret_cast<uint4*>(bb + off) = make_uint4(o0, o1, o2, o3);
    };

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count) {
      pipeline.consumer_wait(smem_pipe_read);
      int const rs = smem_pipe_read.index();
      int const we = ring_write.index() / RingSub;
      uint32_t const*     wp = shared_tensors.smem_Bp.data() + rs * (BLK_N * WPR);
      ElementScale const* sp = shared_tensors.smem_S.data()  + rs * ((BLK_N / 2) * SROW2);
      uint8_t* bb = bb_base + we * RingEntryBytes;

      if constexpr (RingSub == 1) {
        if constexpr (kDep) { ring.producer_acquire(ring_write); }
        CUTLASS_PRAGMA_UNROLL
        for (int it = 0; it < (ITEMS + NumTransformThreads - 1) / NumTransformThreads; ++it) {
          int const item = transform_thread_idx + it * NumTransformThreads;
          if (ITEMS % NumTransformThreads != 0 && item >= ITEMS) { break; }
          int const n = item >> 1;
          int const h = item & 1;
          uint4 const w4 = *reinterpret_cast<uint4 const*>(wp + n * WPR + h * 4);
          uint32_t const sc2 = *reinterpret_cast<uint32_t const*>(sp + (n >> 1) * SROW2 + (n & 1) * SPR + h * 2);
          __nv_bfloat162 const s0 = as_bf162(dup_lo16(sc2));   // k = 32h + 0..15
          __nv_bfloat162 const s1 = as_bf162(dup_hi16(sc2));   // k = 32h + 16..31
          store8(bb, n, 32 * h +  0, w4.x, s0, we);
          store8(bb, n, 32 * h +  8, w4.y, s0, we);
          store8(bb, n, 32 * h + 16, w4.z, s1, we);
          store8(bb, n, 32 * h + 24, w4.w, s1, we);
        }
        if constexpr (kDep) { ring.producer_commit(ring_write); }
        ++ring_write;
      }
      else {
        // quarter hand-over: per k32 half, quarter 2h (words x,y) is finished and published before quarter 2h+1 (z,w)
        constexpr int RPT = (BLK_N + NumTransformThreads - 1) / NumTransformThreads;   // rows per thread
        CUTLASS_PRAGMA_UNROLL
        for (int h = 0; h < BLK_K / 32; ++h) {
          uint4 w4[RPT];
          uint32_t sc2[RPT];
          if constexpr (kDep) { ring.producer_acquire(ring_write); }
          CUTLASS_PRAGMA_UNROLL
          for (int it = 0; it < RPT; ++it) {
            int const n = transform_thread_idx + it * NumTransformThreads;
            if (BLK_N % NumTransformThreads == 0 || n < BLK_N) {
              w4[it]  = *reinterpret_cast<uint4 const*>(wp + n * WPR + h * 4);
              sc2[it] = *reinterpret_cast<uint32_t const*>(sp + (n >> 1) * SROW2 + (n & 1) * SPR + h * 2);
              __nv_bfloat162 const s0 = as_bf162(dup_lo16(sc2[it]));
              store8(bb, n, 32 * h + 0, w4[it].x, s0, we);
              store8(bb, n, 32 * h + 8, w4[it].y, s0, we);
            }
          }
          if constexpr (kDep) { ring.producer_commit(ring_write); }
          ++ring_write;
          if constexpr (kDep) { ring.producer_acquire(ring_write); }
          CUTLASS_PRAGMA_UNROLL
          for (int it = 0; it < RPT; ++it) {
            int const n = transform_thread_idx + it * NumTransformThreads;
            if (BLK_N % NumTransformThreads == 0 || n < BLK_N) {
              __nv_bfloat162 const s1 = as_bf162(dup_hi16(sc2[it]));
              store8(bb, n, 32 * h + 16, w4[it].z, s1, we);
              store8(bb, n, 32 * h + 24, w4[it].w, s1, we);
            }
          }
          if constexpr (kDep) { ring.producer_commit(ring_write); }
          ++ring_write;
        }
      }

      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
    }
  }

  CUTLASS_DEVICE void
  transform_tail(RingPipeline ring, RingPipelineState ring_write) {
    if constexpr (kDep) { ring.producer_tail(ring_write); }
  }

  // ---- MMA warps: the dense sm120 loop, B from the ring instead of the TMA stage ----
  template <class FrgTensorC>
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      RingPipeline ring,
      RingPipelineState ring_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors) {
    using namespace cute;
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    clear(accum);

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});     // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_BB.data()), SmemLayoutBB{});   // (BLK_N,BLK_K,RING)

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCrA = thread_mma.partition_fragment_A(sA(_,_,Int<0>{}));                         // (MMA,MMA_M,MMA_K)
    Tensor tCrB = thread_mma.partition_fragment_B(sB(_,_,Int<0>{}));                         // (MMA,MMA_N,MMA_K)

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(as_position_independent_swizzle_tensor(sA));   // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                                            // (CPY,CPY_M,CPY_K)

    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
    auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(thread_idx);
    Tensor tCsB            = smem_thr_copy_B.partition_S(as_position_independent_swizzle_tensor(sB));   // (CPY,CPY_N,CPY_K,RING)
    Tensor tCrB_copy_view  = smem_thr_copy_B.retile_D(tCrB);                                            // (CPY,CPY_N,CPY_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));
    CUTE_STATIC_ASSERT_V(Int<Stages>{} == size<2>(sA));
    CUTE_STATIC_ASSERT_V(Int<RingStages>{} == size<2>(sB));

    auto K_BLOCK_MAX = size<2>(tCrA);
    static_assert(decltype(K_BLOCK_MAX)::value == KBLOCKS, "one k_block per m16n8k16 step");

    // ring cursors: ring_wait = the entry (or quarter) copy_kblock reads next; ring_rel = the next one to release
    auto ring_wait = ring_read;
    auto ring_rel  = ring_read;
    int read_stage = smem_pipe_read.index();
    int read_entry = ring_wait.index() / RingSub;
    auto tCsA_stage = tCsA(_,_,_,read_stage);
    auto tCsB_stage = tCsB(_,_,_,read_entry);

    auto copy_kblock = [&](auto k_block) {
      copy(smem_tiled_copy_A, tCsA_stage(_,_,k_block), tCrA_copy_view(_,_,k_block));
      copy(smem_tiled_copy_B, tCsB_stage(_,_,k_block), tCrB_copy_view(_,_,k_block));
    };
    auto gemm_kblock = [&](auto k_block) {
      cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), accum);
    };

    pipeline.consumer_wait(smem_pipe_read);
    if constexpr (kDep) { ring.consumer_wait(ring_wait); }

    copy_kblock(_0{});
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count) {
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
        if constexpr (decltype(k_block)::value == KBLOCKS - 1) {
          cutlass::arch::NamedBarrier::sync(thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
          pipeline.consumer_release(smem_pipe_read);
          ++smem_pipe_read;
          read_stage = smem_pipe_read.index();
          tCsA_stage = tCsA(_,_,_,read_stage);
          if constexpr (RingSub == 1 && kDep) { ring.consumer_release(ring_rel); ++ring_rel; }
          ++ring_wait;
          read_entry = ring_wait.index() / RingSub;
          tCsB_stage = tCsB(_,_,_,read_entry);
          pipeline.consumer_wait(smem_pipe_read);
          if constexpr (kDep) { ring.consumer_wait(ring_wait); }
        }
        else if constexpr (RingSub > 1) {
          ++ring_wait;
          if constexpr (kDep) { ring.consumer_wait(ring_wait); }
        }
        copy_kblock(k_block_next);
        // quarter mode: the quarter read by the previous copy_kblock is in registers; free it for the transform warps
        if constexpr (RingSub > 1 && kDep) { ring.consumer_release(ring_rel); ++ring_rel; }
        gemm_kblock(k_block);
      });
    }

    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
      auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
      if constexpr (decltype(k_block)::value == KBLOCKS - 1) {
        cutlass::arch::NamedBarrier::sync(thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        pipeline.consumer_release(smem_pipe_read);
        ++smem_pipe_read;
        if constexpr (kDep) { ring.consumer_release(ring_rel); ++ring_rel; }   // the entry, or its last quarter
      }
      else {
        if constexpr (RingSub > 1) {
          ++ring_wait;
          if constexpr (kDep) { ring.consumer_wait(ring_wait); }
        }
        copy_kblock(k_block_next);
        if constexpr (RingSub > 1 && kDep) { ring.consumer_release(ring_rel); ++ring_rel; }
      }
      gemm_kblock(k_block);
    });
  }

  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline, PipelineState, int) {}
};

} // namespace cutlass::gemm::collective
