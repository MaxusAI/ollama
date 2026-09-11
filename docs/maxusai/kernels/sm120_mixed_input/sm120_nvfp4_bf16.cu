// sm120_nvfp4_bf16.cu
//
// Instantiation of the mixed-input collective in sm120_mma_tma_mixed_input.hpp:
//   D[M][N] (bf16) = alpha * A[M][K] (bf16) * dequant(Bp, Sp)[N][K]^T
// with Bp/Sp the fragment-native packed NVFP4 weight and e4m3 scale tensors described in the header,
// and alpha the optional per-tensor global scale (applied in the epilogue, fp32).
//
// -DNVFP4_TRANSFORM_WARPS=T (T = 1..3) selects the transform-warp variant instead
// (sm120_mma_tma_transform.hpp + sm120_gemm_tma_ws_cooperative_transform.hpp): Bp is then the MLX-native
// uint32 [N][K/8] tensor and Sp the row-pair bf16 scale tensor [N/2][K/8]; sm120_nvfp4_layout() returns 1.
// Further knobs: NVFP4_TW_STAGES (2), NVFP4_TW_RING (2), NVFP4_TW_STATIC_SCHED (0), NVFP4_TW_MODE (2).

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"
#include "cute/atom/mma_traits_sm80.hpp"
#include "cute/atom/copy_traits_sm75.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

#ifndef NVFP4_TRANSFORM_WARPS
#define NVFP4_TRANSFORM_WARPS 0
#endif
#if NVFP4_TRANSFORM_WARPS > 0
#include "sm120_mma_tma_transform.hpp"
#include "sm120_gemm_tma_ws_cooperative_transform.hpp"
#else
#include "sm120_mma_tma_mixed_input.hpp"
#endif

using namespace cute;

namespace {

using ElementA   = cutlass::bfloat16_t;
using ElementB   = uint32_t;                   // packed words
using ElementC   = cutlass::bfloat16_t;
using ElementD   = cutlass::bfloat16_t;
using ElementAcc = float;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;   // nominal; the collective builds its own (N/8, K) descriptor
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignC = 8, AlignD = 8;

#ifndef TILE_M
#define TILE_M 128
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 64
#endif
#ifndef STAGES_OVERRIDE
#define STAGES_OVERRIDE 0
#endif

using TileShape    = Shape<Int<TILE_M>, Int<TILE_N>, Int<TILE_K>>;
using ClusterShape = Shape<_1,_1,_1>;

#ifndef NVFP4_TW_NOSMEM_EPI
#define NVFP4_TW_NOSMEM_EPI 0
#endif
#if NVFP4_TW_NOSMEM_EPI
// Smem-free epilogue (register -> gmem stores, what CUTLASS's sm90 builder emits for NoSmemWarpSpecialized;
// the sm120 builder does not offer it): frees the TMA epilogue's 12 KB so a THIRD TMA stage fits next to the ring.
using CollectiveEpilogue = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::DefaultEpilogue<
      ElementC,
      cutlass::gemm::TagToStrideC_t<LayoutCTag>,
      cutlass::gemm::TagToStrideC_t<LayoutDTag>,
      cutlass::epilogue::thread::LinearCombination<ElementD, 1, ElementAcc, ElementAcc,
          cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, ElementC>,
      cutlass::gemm::EpilogueDefault>>;
#else
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    ElementC, LayoutCTag, AlignC,
    ElementD, LayoutDTag, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;
#endif

using PermTileM     = decltype(cute::min(size<0>(TileShape{}), _128{}));
using PermTileN     = decltype(cute::min(size<1>(TileShape{}),  _32{}));
using AtomLayoutMNK = Layout<Shape<_4,_2,_1>>;

using TiledMma = decltype(make_tiled_mma(
    MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
    AtomLayoutMNK{},
    Tile<PermTileM, PermTileN, _16>{}));

using SmemLayoutAtomA = cute::conditional_t<(TILE_K % 64 == 0), UMMA::Layout_K_SW128_Atom<ElementA>, UMMA::Layout_K_SW64_Atom<ElementA>>;
using SmemCopyAtomA   = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
using GmemTiledCopyA  = SM90_TMA_LOAD;
#if NVFP4_TRANSFORM_WARPS > 0
// B-side atoms describe the bf16 B RING the transform warps fill: the dense kernel's SW128 K-major atom + ldmatrix.
using SmemLayoutAtomB = UMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>;
using SmemCopyAtomB   = Copy_Atom<SM75_U32x4_LDSM_N, cutlass::bfloat16_t>;
#else
// B-side atoms are not used by the mixed-input collective (it owns the packed B and scale paths).
using SmemLayoutAtomB = Layout<Shape<_8,_64>, Stride<_64,_1>>;
using SmemCopyAtomB   = Copy_Atom<DefaultCopy, ElementB>;
#endif
using GmemTiledCopyB  = SM90_TMA_LOAD;

// Stage budget: A bf16 tile + packed B words + scale bytes per stage; epilogue smem carved out.
#ifndef NVFP4_SCALE_BF16
#define NVFP4_SCALE_BF16 0
#endif
constexpr int kScaleBytes = NVFP4_SCALE_BF16 ? 2 : 1;
constexpr int kStageBytes = TILE_M * TILE_K * 2 + (TILE_N / 8) * TILE_K * 4 + (TILE_N / 8) * (TILE_K / 32) * 16 * kScaleBytes;
constexpr int kEpilogueBytes = static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage));
constexpr int kReserve = 256;    // pipeline barriers + alignment slack (the SharedStorage static_assert below is the real guard)
constexpr int StagesAuto = (cutlass::gemm::collective::detail::sm120_smem_capacity_bytes - kEpilogueBytes - kReserve) / kStageBytes;
constexpr int Stages = STAGES_OVERRIDE > 0 ? STAGES_OVERRIDE : StagesAuto;
static_assert(Stages >= 2, "not enough smem for 2 stages at this tile shape");

constexpr uint32_t SchedulerPipelineStageCount = 2;
#if NVFP4_TRANSFORM_WARPS > 0
#ifndef NVFP4_TW_STAGES
#define NVFP4_TW_STAGES 2
#endif
#ifndef NVFP4_TW_RING
#define NVFP4_TW_RING 2
#endif
#ifndef NVFP4_TW_STATIC_SCHED
#define NVFP4_TW_STATIC_SCHED 0
#endif
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeTransformSm120<SchedulerPipelineStageCount>;
using DispatchPolicy = cutlass::gemm::MainloopSm120TmaWarpSpecializedTransform<NVFP4_TW_STAGES, NVFP4_TW_RING, NVFP4_TRANSFORM_WARPS, SchedulerPipelineStageCount, ClusterShape, KernelSchedule>;
using SchedulerTag   = cute::conditional_t<(NVFP4_TW_STATIC_SCHED != 0), cutlass::gemm::StaticPersistentScheduler, void>;
#else
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeSm120<SchedulerPipelineStageCount>;
using DispatchPolicy = cutlass::gemm::MainloopSm120TmaWarpSpecializedMixedInput<Stages, SchedulerPipelineStageCount, ClusterShape, KernelSchedule>;
using SchedulerTag   = void;
#endif

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, cutlass::gemm::TagToStrideA_t<LayoutATag>,
    ElementB, cutlass::gemm::TagToStrideB_t<LayoutBTag>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity>;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, SchedulerTag>;
using Gemm       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
static_assert(sizeof(typename GemmKernel::SharedStorage) <= cutlass::gemm::collective::detail::sm120_smem_capacity_bytes, "smem budget exceeded");

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

void*  g_workspace      = nullptr;
size_t g_workspace_size = 0;

} // namespace

extern "C" {

int sm120_nvfp4_gemm(const void* A, const void* Bp, const void* Sp, void* D, int M, int N, int K, float alpha,
                     void* stream_v, int raster, int swizzle, char* err, int errlen) {
  cudaStream_t stream = static_cast<cudaStream_t>(stream_v);
  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {static_cast<const ElementA*>(A), sA, static_cast<const uint32_t*>(Bp), static_cast<const typename CollectiveMainloop::ElementScale*>(Sp)},
    {{alpha, 0.0f}, static_cast<const ElementC*>(D), sC, static_cast<ElementD*>(D), sD}
  };
  // tile scheduler knobs (runtime): raster 0=Heuristic 1=AlongM 2=AlongN; swizzle = max_swizzle_size (1 = off)
  using RO = cutlass::gemm::kernel::detail::RasterOrderOptions;
  args.scheduler.raster_order = raster == 1 ? RO::AlongM : raster == 2 ? RO::AlongN : RO::Heuristic;
  args.scheduler.max_swizzle_size = swizzle > 0 ? swizzle : 1;
  Gemm gemm;
  cutlass::Status st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) { snprintf(err, errlen, "can_implement: %s", cutlassGetStatusString(st)); return 1; }
  size_t ws = Gemm::get_workspace_size(args);
  if (ws > g_workspace_size) {
    if (g_workspace) cudaFree(g_workspace);
    if (cudaMalloc(&g_workspace, ws) != cudaSuccess) { snprintf(err, errlen, "cudaMalloc workspace %zu", ws); return 4; }
    g_workspace_size = ws;
  }
  st = gemm.initialize(args, g_workspace, stream);
  if (st != cutlass::Status::kSuccess) { snprintf(err, errlen, "initialize: %s", cutlassGetStatusString(st)); return 2; }
  st = gemm.run(stream);
  if (st != cutlass::Status::kSuccess) { snprintf(err, errlen, "run: %s", cutlassGetStatusString(st)); return 3; }
  return 0;
}

#if NVFP4_TRANSFORM_WARPS > 0
// per-warp-role cycle counters (NVFP4_TW_TIMERS=1 builds only): out[8] as documented in the collective; reset zeroes them
int sm120_nvfp4_tw_timers(unsigned long long* out, int reset) {
#if NVFP4_TW_TIMERS
  if (cudaMemcpyFromSymbol(out, cutlass::gemm::collective::nvfp4_tw_detail::g_tw_timers, 12 * sizeof(unsigned long long)) != cudaSuccess) return 1;
  if (reset) {
    unsigned long long z[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
    if (cudaMemcpyToSymbol(cutlass::gemm::collective::nvfp4_tw_detail::g_tw_timers, z, sizeof(z)) != cudaSuccess) return 2;
  }
  return 0;
#else
  (void) out; (void) reset; return 3;
#endif
}
int sm120_nvfp4_scale_format() { return 1; }   // transform variant: bf16 scales always
int sm120_nvfp4_layout() { return 1; }         // 1: MLX-native Bp[N][K/8] + row-pair bf16 Sp[N/2][K/8]
int sm120_nvfp4_tw_info(int* transform_warps, int* ring, int* mode, int* static_sched, int* load_regs, int* mma_regs, int* ring_k16) {
  *transform_warps = NVFP4_TRANSFORM_WARPS; *ring = NVFP4_TW_RING; *mode = NVFP4_TW_MODE; *static_sched = NVFP4_TW_STATIC_SCHED;
  *load_regs = NVFP4_TW_LOAD_REGS; *mma_regs = NVFP4_TW_MMA_REGS; *ring_k16 = NVFP4_TW_RING_K16 + 2 * NVFP4_TW_NOSMEM_EPI;   // bit 1: smem-free epilogue
  return 0;
}
#else
int sm120_nvfp4_scale_format() { return NVFP4_SCALE_BF16; }   // 0: e4m3 bytes, 1: bf16
int sm120_nvfp4_layout() { return 0; }         // 0: fragment-native Bp[N/8][K] + Sp[N/8][K/2]
#endif

int sm120_nvfp4_gemm_info(int* stages, int* tile_m, int* tile_n, int* tile_k, int* smem_bytes) {
#if NVFP4_TRANSFORM_WARPS > 0
  *stages = NVFP4_TW_STAGES;
#else
  *stages = Stages;
#endif
  *tile_m = TILE_M; *tile_n = TILE_N; *tile_k = TILE_K;
  *smem_bytes = static_cast<int>(sizeof(typename GemmKernel::SharedStorage));
  return 0;
}

} // extern "C"
