// sm120_mma_tma_mixed_input.hpp
//
// A mixed-input derivative of CUTLASS 4.8.0's sm120 TMA warp-specialised mainloop
// (include/cutlass/gemm/collective/sm120_mma_tma.hpp): bf16 activations (A) times
// NVFP4 weights (B: e2m1 nibbles with one e4m3 scale per 16 consecutive K values),
// accumulated in fp32 on the Ampere-era per-warp tensor core instruction
// (SM80_16x8x16_F32BF16BF16F32_TN), which is the only bf16 MMA an sm120 part has.
//
// What is kept from the dense mainloop, verbatim in structure:
//   * the TMA producer / mma.sync consumer split (one elected producer thread issues
//     all TMA loads, consumer warps do ldmatrix + mma), PipelineTmaAsync, the
//     per-k-block register double buffering, and the named-barrier stage release.
//   * the A operand path: SW128 K-major smem atom, ldmatrix x4, MMA fragment partition.
// What is added:
//   * a third TMA stream for the e4m3 scales, and a packed uint32 stream for B.
//   * B and the scales use a FRAGMENT-NATIVE ("pre-shuffled", Marlin-style) layout so a
//     lane's whole 8-value contribution to a k32 block is one 32-bit word; see below.
//   * an exact register-side dequantiser between the smem->register copy and the MMA:
//     e2m1 * e4m3 needs at most 6 significant bits, so bf16 holds the product exactly;
//     the only rounding anywhere is the fp32 accumulation and the bf16 output.
// What is deliberately NOT supported (prototype): N % TileN != 0, K % TileK != 0,
// the affine (int4 + bias) modes, and multicast clusters.
//
// Packed layouts (host-side repack from MLX's native uint32 [N, K/8] / uint8 [N, K/16]
// is a pure permutation; see bench_sm120.py::preshuffle_*):
//   Bp[N/8][K] uint32:  word (nt, kb*32 + lane) holds, for n = 8*nt + lane/4 and c = lane%4,
//       nibble j (bits 4j..4j+3) = code of B[n][32*kb + kk(c, j)],
//       kk(c, j) = 16*(j/4) + (j%4 < 2 ? 2c + j%4 : 2c + 8 + (j%4 - 2)),
//     i.e. nibbles 0-3 are the lane's B fragment for the first k16 step of the block
//     (MMA slots k=2c, 2c+1, 2c+8, 2c+9) and nibbles 4-7 the second step.
//   Sp[N/8][K/2] uint8:  byte (nt, kb*16 + (n%8)*2 + s) = e4m3 scale of B[n][32*kb + 16*s .. +16).

#pragma once

// NVFP4_DEQUANT_MODE: 2 = real kernel (default); 1 and 0 strip the scale multiply / the whole
// conversion and exist only to attribute time (their output is wrong by construction).
#ifndef NVFP4_DEQUANT_MODE
#define NVFP4_DEQUANT_MODE 2
#endif
// NVFP4_SCALE_BF16: 0 = scales arrive as e4m3 bytes and are decoded in registers (exact bit trick);
//                   1 = scales arrive pre-converted to bf16 (exact; host-side repack) - saves 4 ALU ops per fragment.
#ifndef NVFP4_SCALE_BF16
#define NVFP4_SCALE_BF16 0
#endif
// NVFP4_CONVERT_X8: 1 = at every even k-block convert the lane's whole 32-bit word (both k16 steps of the
//                   k32 block) with the x8 table converter and load both bf16 scales with one 32-bit load.
//                   Halves the packed-word loads and amortises the permutes. Requires NVFP4_SCALE_BF16=1.
#ifndef NVFP4_CONVERT_X8
#define NVFP4_CONVERT_X8 0
#endif
#if NVFP4_CONVERT_X8 && !NVFP4_SCALE_BF16
#error "NVFP4_CONVERT_X8 requires NVFP4_SCALE_BF16=1"
#endif

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

template <int Stages_, int SchedulerPipelineStageCount_, class ClusterShape_, class KernelSchedule_>
struct MainloopSm120TmaWarpSpecializedMixedInput {
  constexpr static int Stages = Stages_;
  constexpr static int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule_;
  constexpr static int PipelineAsyncMmaStages = 0;
  using ArchTag = arch::Sm120;
};

} // namespace cutlass::gemm

namespace cutlass::gemm::collective {
using namespace cute;

namespace nvfp4_detail {

// e4m3 byte -> bf16 bits scaled down by 2^120:  (s<<15) | (eeee<<7) | (mmm<<4).
// Multiplying that bf16 by 2^120 is exact for every code including the subnormal ones
// (e=0 -> m/8 * 2^-6); NaN (0x7F/0xFF) maps to +-480 rather than NaN.
CUTLASS_DEVICE uint32_t e4m3_to_bf16x2_pre(uint32_t b) {
  uint32_t h = ((b << 8) & 0x8000u) | ((b << 4) & 0x07F0u);
  return h | (h << 16);
}

CUTLASS_DEVICE __nv_bfloat162 as_bf162(uint32_t u) {
  __nv_bfloat162 r; *reinterpret_cast<uint32_t*>(&r) = u; return r;
}
CUTLASS_DEVICE uint32_t as_u32(__nv_bfloat162 v) {
  return *reinterpret_cast<uint32_t*>(&v);
}

// 2^120 as bf16x2 (both halves): exponent field 247.
constexpr uint32_t kTwoPow120x2 = 0x7B807B80u;

// e2m1 -> bf16 via CUTLASS's prmt lookup table (numeric_conversion.h, arch-free).
using E2m1x4ToBf16 = cutlass::NumericArrayConverter<cutlass::bfloat16_t, cutlass::float_e2m1_t, 4>;
using E2m1x8ToBf16 = cutlass::NumericArrayConverter<cutlass::bfloat16_t, cutlass::float_e2m1_t, 8>;

CUTLASS_DEVICE uint32_t dup_lo16(uint32_t x) { return __byte_perm(x, x, 0x1010); }   // both halves = low  16 bits
CUTLASS_DEVICE uint32_t dup_hi16(uint32_t x) { return __byte_perm(x, x, 0x3232); }   // both halves = high 16 bits

} // namespace nvfp4_detail

template <
  int Stages,
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
    MainloopSm120TmaWarpSpecializedMixedInput<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>,
    TileShape_, ElementA_, StrideA_, ElementB_, StrideB_, TiledMma_,
    GmemTiledCopyA_, SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_,
    GmemTiledCopyB_, SmemLayoutAtomB_, SmemCopyAtomB_, TransformB_> {

  using DispatchPolicy = MainloopSm120TmaWarpSpecializedMixedInput<Stages, SchedulerPipelineStageCount, ClusterShape, KernelScheduleType>;
  using TileShape = TileShape_;
  using ElementA  = ElementA_;                 // bf16
  using StrideA   = StrideA_;
  using ElementB  = ElementB_;                 // uint32_t: 8 packed e2m1 codes, fragment-native order
  using StrideB   = StrideB_;                  // kept for the kernel layer; the TMA descriptor is built from (N/8, K)
  using ElementScale = cute::conditional_t<(NVFP4_SCALE_BF16 != 0), uint16_t, uint8_t>;   // e4m3 bits or bf16 bits
  using TiledMma  = TiledMma_;
  using ElementBMma = typename TiledMma::ValTypeB;   // bf16
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using RuntimeDataTypeA = void*;
  using RuntimeDataTypeB = void*;

  static constexpr int ThreadCount = size(TiledMma{});

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState  = typename cutlass::PipelineState<DispatchPolicy::Stages>;

  static constexpr int NumProducerThreadEvents = 1;

  static constexpr int BLK_M = size<0>(TileShape{});
  static constexpr int BLK_N = size<1>(TileShape{});
  static constexpr int BLK_K = size<2>(TileShape{});
  static_assert(BLK_N % 8 == 0 && BLK_K % 32 == 0, "packed layout needs N tile % 8 and K tile % 32");
  static constexpr int NT = BLK_N / 8;          // 8-wide n-tiles per CTA tile
  static constexpr int KB = BLK_K / 32;         // k32 blocks per CTA k-tile
  static constexpr int SROW = KB * 16;          // scale elements per n-tile per k-tile

  static_assert(cute::is_same_v<ElementB, uint32_t>, "ElementB must be the packed uint32 word type");
  static_assert(cute::is_same_v<ElementBMma, cutlass::bfloat16_t>, "MMA B type must be bf16");
  static_assert(cute::is_same_v<ElementA, cutlass::bfloat16_t>, "A must be bf16");
  static_assert(decltype(size<2>(typename TiledMma::AtomShape_MNK{}))::value == 16, "expects the m16n8k16 bf16 atom");

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert(not cute::is_void_v<SmemCopyAtomA>, "SM120 mainloop must specify a copy atom for A operand smem->rmem reads.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  // Packed B words: (NT, BLK_K, PIPE) row-major, no swizzle needed (each warp reads 32 consecutive words).
  using SmemLayoutB = decltype(make_layout(
      make_shape(Int<NT>{}, Int<BLK_K>{}, Int<DispatchPolicy::Stages>{}),
      make_stride(Int<BLK_K>{}, _1{}, Int<NT * BLK_K>{})));
  // Scales: (NT, SROW, PIPE) row-major elements (uint8 e4m3 or uint16 bf16).
  using SmemLayoutS = decltype(make_layout(
      make_shape(Int<NT>{}, Int<SROW>{}, Int<DispatchPolicy::Stages>{}),
      make_stride(Int<SROW>{}, _1{}, Int<NT * SROW>{})));

  static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                not cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operands from rmem for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD>, "GmemTiledCopyA - only SM90_TMA_LOAD (no multicast) in this prototype.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD>, "GmemTiledCopyB - only SM90_TMA_LOAD (no multicast) in this prototype.");
  static_assert(cute::size(ClusterShape{}) == 1, "no clusters in this prototype");

  using TmaInternalElementA = uint_bit_t<sizeof_bits_v<ElementA>>;

  static constexpr uint32_t TmaTransactionBytesMK = static_cast<uint32_t>(
      cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutA{})) * sizeof_bits<ElementA>::value));
  static constexpr uint32_t TmaTransactionBytesNK = static_cast<uint32_t>(
      NT * BLK_K * sizeof(uint32_t) + NT * SROW * sizeof(ElementScale));
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      alignas(1024) cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_A;
      alignas(128)  cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutB>> smem_B;
      alignas(128)  cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutS>> smem_S;
    } tensors;
    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    alignas(16) PipelineStorage pipeline_storage;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  struct Arguments {
    ElementA const* ptr_A{nullptr};
    StrideA dA{};
    uint32_t const* ptr_B{nullptr};   // Bp[N/8][K]
    ElementScale const* ptr_S{nullptr};   // Sp[N/8][K/2] (elements)
  };

  using StrideBp = Stride<int64_t, Int<1>, int64_t>;   // (N/8, K, L) row-major words
  using StrideSp = Stride<int64_t, Int<1>, int64_t>;   // (N/8, K/2, L) row-major bytes

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
        SmemLayoutB{}(_,_,0),
        make_shape(Int<NT>{}, Int<BLK_K>{}),
        _1{}));
    using TMA_S = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(static_cast<ElementScale const*>(nullptr), make_shape(int32_t(0), int32_t(0), int32_t(0)), StrideSp{}),
        SmemLayoutS{}(_,_,0),
        make_shape(Int<NT>{}, Int<SROW>{}),
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

    int32_t const n8 = N / 8;
    Tensor tensor_b = make_tensor(args.ptr_B, make_layout(make_shape(n8, K, L),
        make_stride(int64_t(K), Int<1>{}, int64_t(n8) * int64_t(K))));
    typename Params::TMA_B tma_load_b = make_tma_copy(
        SM90_TMA_LOAD{}, tensor_b, SmemLayoutB{}(_,_,cute::Int<0>{}),
        make_shape(Int<NT>{}, Int<BLK_K>{}), _1{});

    int32_t const k2 = K / 2;
    Tensor tensor_s = make_tensor(args.ptr_S, make_layout(make_shape(n8, k2, L),
        make_stride(int64_t(k2), Int<1>{}, int64_t(n8) * int64_t(k2))));
    typename Params::TMA_S tma_load_s = make_tma_copy(
        SM90_TMA_LOAD{}, tensor_s, SmemLayoutS{}(_,_,cute::Int<0>{}),
        make_shape(Int<NT>{}, Int<SROW>{}), _1{});

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
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: mixed-input prototype needs N % TileN == 0, K % TileK == 0, L == 1, 16B-aligned A.\n");
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
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N/8,K,L));               // (n8,k,l) words
    Tensor mS_nkl = mainloop_params.tma_load_s.get_tma_tensor(make_shape(N/8,K/2,L));             // (n8,k/2,l) scale elements
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});         // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, make_shape(Int<NT>{}, Int<BLK_K>{}), make_coord(_,_,_));   // (NT,BLK_K,n,k,l)
    Tensor gS_nkl = local_tile(mS_nkl, make_shape(Int<NT>{}, Int<SROW>{}), make_coord(_,_,_));    // (NT,SROW,n,k,l)
    return cute::make_tuple(gA_mkl, gB_nkl, gS_nkl);
  }

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
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (NT,BLK_K,PIPE)
      Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_S.data()), SmemLayoutS{});        // (NT,SROW,PIPE)

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gB_nkl = get<1>(load_inputs);
      Tensor gS_nkl = get<2>(load_inputs);

      auto block_tma_a = mainloop_params.tma_load_a.get_slice(0);
      auto block_tma_b = mainloop_params.tma_load_b.get_slice(0);
      auto block_tma_s = mainloop_params.tma_load_s.get_slice(0);

      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                  // (BLK_M,BLK_K,k)
      Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                  // (NT,BLK_K,k)
      Tensor gS = gS_nkl(_,_,n_coord,_,l_coord);                                                  // (NT,SROW,k)

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

  template <class FrgTensorC, class BlockCoord>
  CUTLASS_DEVICE void
  mma(MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      Params const& mainloop_params,
      [[maybe_unused]] BlockCoord& blk_crd) {
    using namespace cute;
    using namespace nvfp4_detail;
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");

    clear(accum);

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});    // (BLK_M,BLK_K,PIPE)
    uint32_t const* sB_words = shared_tensors.smem_B.data();
    ElementScale const* sS_elems = shared_tensors.smem_S.data();

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_thread_slice(thread_idx);

    // ---- A: identical to the dense mainloop ----
    Tensor tCrA = thread_mma.partition_fragment_A(sA(_,_,Int<0>{}));                         // (MMA,MMA_M,MMA_K)
    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(thread_idx);
    Tensor tCsA            = smem_thr_copy_A.partition_S(as_position_independent_swizzle_tensor(sA));   // (CPY,CPY_M,CPY_K,PIPE)
    Tensor tCrA_copy_view  = smem_thr_copy_A.retile_D(tCrA);                                            // (CPY,CPY_M,CPY_K)

    // ---- B: bf16 fragments produced by the dequantiser ----
    // A dummy bf16 (BLK_N, BLK_K) smem view gives partition_fragment_B the right fragment shape/type.
    Tensor sB_logical = make_tensor(make_smem_ptr(static_cast<ElementBMma*>(nullptr)),
                                    make_layout(make_shape(Int<BLK_N>{}, Int<BLK_K>{}), make_stride(Int<BLK_K>{}, _1{})));
    Tensor tCrB = thread_mma.partition_fragment_B(sB_logical);                               // (MMA,MMA_N,MMA_K) bf16
    // Which (n, k) does fragment (jn, k_block) start at?  Ask CuTe rather than re-derive the tiled-MMA permutation.
    Tensor cB   = make_identity_tensor(make_shape(Int<BLK_N>{}, Int<BLK_K>{}));
    Tensor tCcB = thread_mma.partition_B(cB);                                                // (MMA,MMA_N,MMA_K) coords

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));
    CUTE_STATIC_ASSERT_V(size<0>(tCrB) == Int<4>{});   // m16n8k16 B fragment is 4 bf16 per thread

    constexpr int MMA_N = size<1>(tCrB);
    auto K_BLOCK_MAX = size<2>(tCrA);
    static_assert(decltype(K_BLOCK_MAX)::value == BLK_K / 16, "one k_block per m16n8k16 step");

    int const lane = thread_idx & 31;
    // This thread's n-tiles are nt(jn) = nt0 + NT_STRIDE*jn: with AtomLayoutMNK 4x2x1 and the Tile<128,32,16>
    // permutation the two N-warps interleave 8-wide n-tiles (n = 32t + 16j + 8wn + [0,8), jn = j + 2t), so the
    // stride is a compile-time 2 and every smem load below takes an immediate offset. NVFP4_CHECK_NT traps if
    // the TiledMma's partition disagrees (used once in the quick validation build).
    static constexpr int NT_STRIDE = 2;
    int const nt0   = get<0>(tCcB(_0{}, _0{}, _0{})) >> 3;
    int const b_off0 = nt0 * BLK_K + lane;                  // word index of (nt0, kb=0, lane)
    int const s_off0 = nt0 * SROW + ((lane >> 2) << 1);     // element index of (nt0, kb=0, n%8, s=0)
#ifdef NVFP4_CHECK_NT
    CUTLASS_PRAGMA_UNROLL
    for (int jn = 0; jn < MMA_N; ++jn) {
      if ((get<0>(tCcB(_0{}, jn, _0{})) >> 3) != nt0 + NT_STRIDE * jn) { __trap(); }
    }
#endif

    int read_stage = smem_pipe_read.index();
    auto tCsA_stage = tCsA(_,_,_,read_stage);

    [[maybe_unused]] __nv_bfloat162 const two120 = as_bf162(kTwoPow120x2);

    // k_block arrives as cute::Int<> from for_each or as a plain int (k_block_next); both fold after unrolling.
    auto dequant_kblock = [&](auto k_block, int stage) {
      int const kbi = static_cast<int>(k_block);
      int const kb = kbi >> 1;
      int const s  = kbi & 1;
      uint32_t const*     wbase = sB_words + stage * (NT * BLK_K) + b_off0 + kb * 32;
      ElementScale const* sbase = sS_elems + stage * (NT * SROW) + s_off0 + kb * 16 + s;
#if NVFP4_CONVERT_X8
      if (s != 0) { return; }   // the even k-block already produced both steps of this k32 block
      CUTLASS_PRAGMA_UNROLL
      for (int jn = 0; jn < MMA_N; ++jn) {
        uint32_t w   = wbase[jn * (NT_STRIDE * BLK_K)];
        uint32_t sc2 = *reinterpret_cast<uint32_t const*>(sbase + jn * (NT_STRIDE * SROW));   // (s=0, s=1) bf16 pair
        cutlass::Array<cutlass::float_e2m1_t, 8> src8;
        *reinterpret_cast<uint32_t*>(&src8) = w;
        cutlass::Array<cutlass::bfloat16_t, 8> v8 = E2m1x8ToBf16::convert(src8);
        uint32_t const* vv = reinterpret_cast<uint32_t const*>(&v8);
        __nv_bfloat162 sc0 = as_bf162(dup_lo16(sc2)), sc1 = as_bf162(dup_hi16(sc2));
        uint32_t u0 = as_u32(__hmul2(as_bf162(vv[0]), sc0)), u1 = as_u32(__hmul2(as_bf162(vv[1]), sc0));
        uint32_t u2 = as_u32(__hmul2(as_bf162(vv[2]), sc1)), u3 = as_u32(__hmul2(as_bf162(vv[3]), sc1));
        // the 4-bf16 fragment slice is contiguous in registers: store the two 32-bit words directly
        Tensor f0 = recast<uint32_t>(tCrB(_, jn, kbi));
        Tensor f1 = recast<uint32_t>(tCrB(_, jn, kbi + 1));
        f0(_0{}) = u0; f0(_1{}) = u1;
        f1(_0{}) = u2; f1(_1{}) = u3;
      }
      return;
#endif
      CUTLASS_PRAGMA_UNROLL
      for (int jn = 0; jn < MMA_N; ++jn) {
        uint32_t w  = wbase[jn * (NT_STRIDE * BLK_K)];
        uint32_t sc = sbase[jn * (NT_STRIDE * SROW)];
        uint32_t h  = (w >> (16 * s)) & 0xFFFFu;          // 4 nibbles = this step's B fragment
#if NVFP4_DEQUANT_MODE == 0
        // timing probe only: no conversion at all (wrong numerics), same loads
        uint32_t u01 = h | (h << 16), u23 = u01 ^ sc;
#else
        cutlass::Array<cutlass::float_e2m1_t, 4> src;
        *reinterpret_cast<uint16_t*>(&src) = static_cast<uint16_t>(h);
        cutlass::Array<cutlass::bfloat16_t, 4> v = E2m1x4ToBf16::convert(src);
        uint32_t const* vv = reinterpret_cast<uint32_t const*>(&v);
#if NVFP4_DEQUANT_MODE == 1
        // timing probe only: conversion but no scale (wrong numerics)
        uint32_t u01 = vv[0] ^ sc, u23 = vv[1];
#else
#if NVFP4_SCALE_BF16
        __nv_bfloat162 scale = as_bf162(sc | (sc << 16));                                // bf16 scale, both halves
#else
        __nv_bfloat162 scale = __hmul2(as_bf162(e4m3_to_bf16x2_pre(sc)), two120);     // exact e4m3 value
#endif
        __nv_bfloat162 v01 = __hmul2(as_bf162(vv[0]), scale);                            // exact products
        __nv_bfloat162 v23 = __hmul2(as_bf162(vv[1]), scale);
        uint32_t u01 = as_u32(v01), u23 = as_u32(v23);
#endif
#endif
        Tensor f = recast<uint32_t>(tCrB(_, jn, k_block));
        f(_0{}) = u01; f(_1{}) = u23;
      }
    };

    auto copy_kblock = [&](auto k_block) {
      copy(smem_tiled_copy_A, tCsA_stage(_,_,k_block), tCrA_copy_view(_,_,k_block));
      dequant_kblock(k_block, read_stage);
    };

    auto gemm_kblock = [&](auto k_block) {
      cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), accum);
    };

    pipeline.consumer_wait(smem_pipe_read);

    copy_kblock(_0{});
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count) {
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
        if (k_block == K_BLOCK_MAX - 1) {
          cutlass::arch::NamedBarrier::sync(thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
          pipeline.consumer_release(smem_pipe_read);
          ++smem_pipe_read;
          read_stage = smem_pipe_read.index();
          tCsA_stage = tCsA(_,_,_,read_stage);
          pipeline.consumer_wait(smem_pipe_read);
        }
        copy_kblock(k_block_next);
        gemm_kblock(k_block);
      });
    }

    for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
      auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
      if (k_block == K_BLOCK_MAX - 1) {
        cutlass::arch::NamedBarrier::sync(thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
        pipeline.consumer_release(smem_pipe_read);
        ++smem_pipe_read;
      }
      if (k_block_next > 0) {
        copy_kblock(k_block_next);
      }
      gemm_kblock(k_block);
    });
  }

  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline, PipelineState, int) {}
};

} // namespace cutlass::gemm::collective
