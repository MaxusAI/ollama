// sm120_bf16_dense.cu
//
// Hand instantiation of CUTLASS's sm120 TMA warp-specialised mainloop
// (include/cutlass/gemm/collective/sm120_mma_tma.hpp) with the Ampere bf16
// mma.sync atom SM80_16x8x16_F32BF16BF16F32_TN.
//
// Why by hand: CollectiveBuilder<arch::Sm120, OpClassTensorOp, ...> static-asserts
// that both operands are F8F6F4 elements (sm120_mma_builder.inl:80-81 and :115) and
// rr_op_selector_sm120 (cute/arch/mma_sm120.hpp:3253) only ever returns the
// kind::f8f6f4 atom. There is no bf16 collective for sm120 in CUTLASS 4.8, so this
// file replicates the builder body with a bf16 TiledMma to measure what a "tuned
// dense collective" ceiling would be for a loader-swap design on this chip.
//
// D[M][N] (row-major bf16) = A[M][K] (row-major bf16) * B[N][K]^T (K-major bf16)

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
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"
#include "cute/atom/mma_traits_sm80.hpp"
#include "cute/atom/copy_traits_sm75.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

using namespace cute;

namespace {

using ElementA   = cutlass::bfloat16_t;
using ElementB   = cutlass::bfloat16_t;
using ElementC   = cutlass::bfloat16_t;
using ElementD   = cutlass::bfloat16_t;
using ElementAcc = float;
using LayoutATag = cutlass::layout::RowMajor;     // A[M][K], K contiguous
using LayoutBTag = cutlass::layout::ColumnMajor;  // B[N][K], K contiguous (torch Linear weight)
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

using TileShape    = Shape<Int<TILE_M>, Int<TILE_N>, Int<TILE_K>>;
using ClusterShape = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    ElementC, LayoutCTag, AlignC,
    ElementD, LayoutDTag, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

// ---- replicate sm120_mma_builder.inl:90-168 with a bf16 atom ----
using PermTileM     = decltype(cute::min(size<0>(TileShape{}), _128{}));
using PermTileN     = decltype(cute::min(size<1>(TileShape{}),  _32{}));
using AtomLayoutMNK = Layout<Shape<_4,_2,_1>>;   // cooperative schedule: 8 MMA warps

using TiledMma = decltype(make_tiled_mma(
    MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
    AtomLayoutMNK{},
    Tile<PermTileM, PermTileN, _16>{}));

// 128B-swizzled K-major atom is 8 x 64 bf16; a K=32 stage needs the 64B one (8 x 32). Both are TMA + ldmatrix friendly.
using SmemLayoutAtomA = cute::conditional_t<(TILE_K % 64 == 0), UMMA::Layout_K_SW128_Atom<ElementA>, UMMA::Layout_K_SW64_Atom<ElementA>>;
using SmemLayoutAtomB = cute::conditional_t<(TILE_K % 64 == 0), UMMA::Layout_K_SW128_Atom<ElementB>, UMMA::Layout_K_SW64_Atom<ElementB>>;
using SmemCopyAtomA   = Copy_Atom<SM75_U32x4_LDSM_N, ElementA>;
using SmemCopyAtomB   = Copy_Atom<SM75_U32x4_LDSM_N, ElementB>;
using GmemTiledCopyA  = SM90_TMA_LOAD;
using GmemTiledCopyB  = SM90_TMA_LOAD;

using MainloopPipelineStorage = typename cutlass::PipelineTmaUmmaAsync<1>::SharedStorage;
constexpr int Stages = cutlass::gemm::collective::detail::sm100_compute_stage_count_or_override<
    cutlass::gemm::collective::detail::sm120_smem_capacity_bytes, ElementA, ElementB, TileShape, MainloopPipelineStorage>(
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>{});
static_assert(Stages >= 2, "not enough smem for 2 stages at this tile shape");

constexpr uint32_t SchedulerPipelineStageCount = 2;
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeSm120<SchedulerPipelineStageCount>;
using DispatchPolicy = cutlass::gemm::MainloopSm120TmaWarpSpecialized<Stages, SchedulerPipelineStageCount, ClusterShape, KernelSchedule>;

using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy, TileShape,
    ElementA, cutlass::gemm::TagToStrideA_t<LayoutATag>,
    ElementB, cutlass::gemm::TagToStrideB_t<LayoutBTag>,
    TiledMma,
    GmemTiledCopyA, SmemLayoutAtomA, SmemCopyAtomA, cute::identity,
    GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity>;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

void*  g_workspace      = nullptr;
size_t g_workspace_size = 0;

} // namespace

extern "C" {

int sm120_bf16_gemm(const void* A, const void* B, void* D, int M, int N, int K, void* stream_v, int raster, int swizzle, char* err, int errlen) {
  cudaStream_t stream = static_cast<cudaStream_t>(stream_v);
  StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {static_cast<const ElementA*>(A), sA, static_cast<const ElementB*>(B), sB},
    {{1.0f, 0.0f}, static_cast<const ElementC*>(D), sC, static_cast<ElementD*>(D), sD}
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

int sm120_bf16_gemm_info(int* stages, int* tile_m, int* tile_n, int* tile_k, int* smem_bytes) {
  *stages = Stages; *tile_m = TILE_M; *tile_n = TILE_N; *tile_k = TILE_K;
  *smem_bytes = static_cast<int>(sizeof(typename GemmKernel::SharedStorage));
  return 0;
}

} // extern "C"
