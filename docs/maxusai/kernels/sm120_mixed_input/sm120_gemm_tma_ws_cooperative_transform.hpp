/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// sm120_gemm_tma_ws_cooperative_transform.hpp
//
// Kernel layer for the transform-warp mixed-input mainloop (sm120_mma_tma_transform.hpp): a copy of
// CUTLASS 4.8.0's include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp (which
// is what sm120 uses) with these changes, all marked "TW:":
//   * dispatch on KernelTmaWarpSpecializedCooperativeTransformSm120 instead of KernelTmaWarpSpecializedCooperative;
//   * producer warps 4-T .. 3 are TRANSFORM warps (T = CollectiveMainloop::NumTransformWarps): they are extra
//     consumers of the TMA mainloop pipeline and producers of the bf16 B ring (PipelineAsync);
//   * the MMA warps consume both pipelines;
//   * warpgroup register reallocation is a knob (NVFP4_TW_LOAD_REGS / NVFP4_TW_MMA_REGS, 0 = leave the
//     launch budget of 168 alone: the dense consumer loop needs 168 and never used the raised cap);
//   * the MainloopAux role is gone; the epilogue source-load warp (2) may not be needed when it is a
//     transform warp (beta must be 0; the kernel traps otherwise).
// A TileSchedulerSelector for StaticPersistentScheduler on Sm120 is added so that T = 3 (which takes the
// CLC scheduler's warp 1) can use the static persistent scheduler.

#pragma once

#ifndef NVFP4_TW_LOAD_REGS
#define NVFP4_TW_LOAD_REGS 0
#endif
#ifndef NVFP4_TW_MMA_REGS
#define NVFP4_TW_MMA_REGS 0
#endif

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "cute/tensor.hpp"
#include "cutlass/trace.h"
#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include "cutlass/arch/grid_dependency_control.h"
#include "sm120_mma_tma_transform.hpp"

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::kernel {

// TW: static persistent scheduler for Sm120 (CUTLASS only maps it for Sm100/Sm107).
namespace detail {
template <class TileShape, class ClusterShape, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<StaticPersistentScheduler, arch::Sm120, TileShape, ClusterShape, SchedulerPipelineStageCount> {
  using Scheduler = StaticPersistentTileScheduler100;
};
} // namespace detail

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileSchedulerTag_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileSchedulerTag_,
  cute::enable_if_t<cute::is_base_of_v<KernelTmaWarpSpecializedCooperativeTransformSm120Base, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>   // TW
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(ArchTag::kMinComputeCapability >= 90);

  static constexpr uint32_t TileSchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  using TileSchedulerTag = TileSchedulerTag_;

  using TileScheduler = typename detail::TileSchedulerSelector<
                                          TileSchedulerTag, 
                                          ArchTag, 
                                          TileShape,
                                          ClusterShape
                                          ,TileSchedulerPipelineStageCount
                                          >::Scheduler;

  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  
  // Warp specialization thread count per threadblock
  static constexpr uint32_t NumSchedThreads        = NumThreadsPerWarp;      // 1 warp       
  static constexpr uint32_t NumMMAThreads          = size(TiledMma{});       // 8 warps
  static constexpr uint32_t NumMainloopLoadThreads = NumThreadsPerWarp;      // 1 warp
  static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp;      // 1 warp for C

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;
  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = NumMMAThreads / NumThreadsPerWarpGroup;
  static constexpr uint32_t MaxThreadsPerBlock = NumMMAThreads + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr uint32_t NumFixupBarriers = NumMmaWarpGroups;
  static constexpr uint32_t NumProducerThreads = CollectiveMainloop::NumProducerThreadEvents;
  // TW: transform warps = the last NumTransformWarps warps of the producer warpgroup
  static constexpr uint32_t NumTransformWarps   = CollectiveMainloop::NumTransformWarps;
  static constexpr uint32_t NumTransformThreads = CollectiveMainloop::NumTransformThreads;
  static constexpr uint32_t FirstTransformWarp  = NumWarpsPerWarpGroup - NumTransformWarps;
  using RingPipeline = typename CollectiveMainloop::RingPipeline;
  using RingPipelineState = typename CollectiveMainloop::RingPipelineState;

  /// Register requirement for Load and Math WGs
  static constexpr int RegsPerThread =
    size<0>(TileShape{}) * size<1>(TileShape{}) / NumMMAThreads *
    sizeof(ElementAccumulator) / sizeof(uint32_t);
  
  // Detect if this is SM120 blockscaled kernel which hits high register pressure
  // on smaller tiles (e.g. 256x128 registers per thread)
  template <typename T>
  struct IsSm120BlockScaled : cute::false_type {};
  
  template <int Stages, int SchedStages, class ClusterShape, class KernelSchedule>
  struct IsSm120BlockScaled<MainloopSm120TmaWarpSpecializedBlockScaled<Stages, SchedStages, ClusterShape, KernelSchedule>> 
    : cute::true_type {};
  
  static constexpr bool IsSm120Family = cute::is_same_v<typename DispatchPolicy::ArchTag, arch::Sm120>;
  
  static constexpr bool HeavyRegisterPressure = 
    IsSm120BlockScaled<DispatchPolicy>::value ? (RegsPerThread >= 128) : (RegsPerThread >= 208);
  
  static constexpr uint32_t LoadRegisterRequirement = NVFP4_TW_LOAD_REGS;   // TW: 0 = no reallocation
  static constexpr uint32_t MmaRegisterRequirement  = NVFP4_TW_MMA_REGS;

  // 1 stage ordered sequence between mainloop and epilogue producer load threads
  using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;

  using TileSchedulerPipeline = typename TileScheduler::Pipeline;
  using TileSchedulerPipelineState = typename TileSchedulerPipeline::PipelineState;
  using TileSchedulerStorage = typename TileScheduler::SharedStorage;
  using TileSchedulerThrottlePipeline = typename TileScheduler::ThrottlePipeline;
  using TileSchedulerThrottlePipelineState = typename TileSchedulerThrottlePipeline::PipelineState;
  
  // Kernel level shared memory storage
  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
    } pipelines;

    alignas(16) TileSchedulerStorage scheduler;

    struct TensorStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

#if !NVFP4_TW_RING_EPI
      EpilogueTensorStorage epilogue;
#endif
      MainloopTensorStorage mainloop;   // TW: with NVFP4_TW_RING_EPI the epilogue borrows a ring entry (see epi_tensors)
    } tensors;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
    void* workspace{nullptr};
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    auto problem_shape = args.problem_shape;
    if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>) {
      // swap M/N
      get<0>(problem_shape) = get<1>(args.problem_shape);
      get<1>(problem_shape) = get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    // Get maximum number of clusters that could co-exist on the target device
    int max_active_clusters = args.hw_info.max_active_clusters;
    if (max_active_clusters <= 0) {
      max_active_clusters = 0;
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid max cluster count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the max_active_clusters.");
    }
    else {
      CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid cluster count to " << max_active_clusters);
    }

    KernelHardwareInfo hw_info = args.hw_info;
    hw_info.sm_count = sm_count;
    hw_info.max_active_clusters = max_active_clusters;
  
    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;
    // Precompute the sub tiles numbers in epilogue, pass into tile scheduler.  Therefore it will be used
    // in separate reduction scheme for streamk case, NumEpilogueSubTiles default value is 1, which means
    // subtile will not be used, therefore separate reduction will not be enabled.
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
      problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles
      );

    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, epilogue_workspace),
      hw_info,
      scheduler,
      workspace
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && cute::rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler, args.hw_info);
    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    workspace_size += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);
    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});
    static constexpr uint32_t NumAccumulatorMtxs = 1;

    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, args.problem_shape, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles, NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN ? TileScheduler::RasterOrderOptions::AlongN : TileScheduler::RasterOrderOptions::AlongM;
    return TileScheduler::get_grid_shape(params.scheduler, params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

#  if (defined(__CUDA_ARCH_FEAT_SM90_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) ||\
      CUDA_ARCH_CONDITIONAL_OR_FAMILY(1200) || CUDA_ARCH_CONDITIONAL_OR_FAMILY(1210)\
      )
#    define ENABLE_SM90_KERNEL_LEVEL 1
#  endif

// Any Tensor Op MMA Atom in the ISA is arch conditional.
#if ! defined(ENABLE_SM90_KERNEL_LEVEL)
    CUTE_INVALID_CONTROL_PATH("ERROR : Arch conditional MMA instruction used without targeting appropriate compute capability. Aborting.\n");
#else

    // Preconditions
    static_assert(NumMMAThreads == 256, "Cooperative kernel must have TiledMMA operating using 256 threads.");
    // TW: warp 1 of the producer group is the CLC scheduler warp when the scheduler is dynamic persistent
    static_assert(!IsSchedDynamicPersistent || NumTransformWarps <= 2, "T = 3 needs the static persistent scheduler (warp 1 is the CLC scheduler warp)");
    static_assert(size<0>(TileShape{}) >= 128,
        "Cooperative kernel requires Tile Size to be greater than or equal to 128 along the M-dimension.");

    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    /* In the Cooperative kernel, Consumer0 and Consumer1 collaborate on the same tile */
    enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2
    };
    enum class ProducerWarpRole {
      Mainloop = 0,
      Warp1 = 1,
      Epilogue = 2,
      MainloopAux = 3
    };
    // TW: is this producer warp a transform warp?  (warps FirstTransformWarp .. 3)



    // Kernel level shared memory storage
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    // TW: the epilogue's shared memory: its own carve-out, or (NVFP4_TW_RING_EPI) the bf16 ring entry the MMA
    // warps read last, which mma() leaves un-released until the epilogue's TMA stores have drained it.
    using EpilogueTensorStorage = typename SharedStorage::TensorStorage::EpilogueTensorStorage;
    auto epi_tensors = [&](int ring_entry) -> EpilogueTensorStorage& {
#if NVFP4_TW_RING_EPI
      static_assert(sizeof(EpilogueTensorStorage) <= size_t(CollectiveMainloop::RingEntryBytes), "epilogue smem must fit one ring entry");
      static_assert(alignof(EpilogueTensorStorage) <= 1024, "ring entries are 1024 B aligned");
      return *reinterpret_cast<EpilogueTensorStorage*>(
          reinterpret_cast<uint8_t*>(shared_storage.tensors.mainloop.smem_BB.data()) + ring_entry * CollectiveMainloop::RingEntryBytes);
#else
      (void) ring_entry;
      return shared_storage.tensors.epilogue;
#endif
    };

    int thread_idx = int(threadIdx.x);
    int lane_idx = canonical_lane_idx();
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    int mma_thread_idx = thread_idx % NumMMAThreads;
    auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    bool const is_transform_warp = (warp_group_role == WarpGroupRole::Producer) && (warp_idx_in_warp_group >= int(FirstTransformWarp));   // TW
    int  const transform_thread_idx = thread_idx - int(FirstTransformWarp) * NumThreadsPerWarp;                                           // TW
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    CollectiveEpilogue collective_epilogue(params.epilogue, epi_tensors(0));
    bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();
#if NVFP4_TW_RING_EPI
    if (is_epi_load_needed) { __trap(); }   // TW: a source (beta != 0) would need its own smem; the ring entry is D only
#endif
    // TileScheduler pipeline
    typename TileSchedulerPipeline::Params scheduler_pipeline_params;
    typename TileSchedulerThrottlePipeline::Params scheduler_throttle_pipeline_params;
    if constexpr (IsSchedDynamicPersistent) { 
      if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Warp1) {
        scheduler_pipeline_params.role = TileSchedulerPipeline::ThreadCategory::ProducerConsumer;
      }
      else {
        scheduler_pipeline_params.role = TileSchedulerPipeline::ThreadCategory::Consumer;
      }
      scheduler_pipeline_params.producer_blockid = 0;
      scheduler_pipeline_params.producer_arv_count = 1;
      scheduler_pipeline_params.consumer_arv_count = NumSchedThreads + NumMainloopLoadThreads + NumMMAThreads + NumTransformThreads;   // TW

      if (is_epi_load_needed) {
        scheduler_pipeline_params.consumer_arv_count += NumEpilogueLoadThreads;
      } 
      scheduler_pipeline_params.transaction_bytes = sizeof(typename TileScheduler::CLCResponse);
      
      scheduler_throttle_pipeline_params.producer_arv_count = NumMainloopLoadThreads;
      scheduler_throttle_pipeline_params.consumer_arv_count = NumSchedThreads;
      scheduler_throttle_pipeline_params.dst_blockid = 0;
      scheduler_throttle_pipeline_params.initializing_warp = 3;
      if (warp_group_role == WarpGroupRole::Producer &&
          producer_warp_role == ProducerWarpRole::Warp1) {
        scheduler_throttle_pipeline_params.role =
            TileSchedulerThrottlePipeline::ThreadCategory::Consumer;
      }
      // set role when it is for DMA warp in Mainloop
      else if (warp_group_role == WarpGroupRole::Producer &&
               producer_warp_role == ProducerWarpRole::Mainloop) {
        scheduler_throttle_pipeline_params.role =
            TileSchedulerThrottlePipeline::ThreadCategory::Producer;
      }
    }
    TileSchedulerPipeline scheduler_pipeline(shared_storage.scheduler.pipeline(), scheduler_pipeline_params);
    TileSchedulerPipelineState scheduler_pipe_consumer_state;

    TileSchedulerThrottlePipeline scheduler_throttle_pipeline(shared_storage.scheduler.throttle_pipeline(), scheduler_throttle_pipeline_params);
    TileSchedulerThrottlePipelineState scheduler_pipe_throttle_consumer_state;
    TileSchedulerThrottlePipelineState scheduler_pipe_throttle_producer_state = cutlass::make_producer_start_state<TileSchedulerThrottlePipeline>();

    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Mainloop) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1 || is_transform_warp) {   // TW
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = NumMMAThreads + NumTransformThreads;   // TW: every consumer thread arrives (cluster size 1)
    mainloop_pipeline_params.num_producers = NumProducerThreads;
    mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop.tma, mainloop_pipeline_params, ClusterShape{});

    // TW: bf16 B ring, transform warps -> MMA warps
    typename RingPipeline::Params ring_pipeline_params;
    if (is_transform_warp) {
      ring_pipeline_params.role = RingPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      ring_pipeline_params.role = RingPipeline::ThreadCategory::Consumer;
    }
    ring_pipeline_params.producer_arv_count = NumTransformThreads;
    ring_pipeline_params.consumer_arv_count = NumMMAThreads;
    ring_pipeline_params.dst_blockid = 0;
    RingPipeline ring_pipeline(shared_storage.pipelines.mainloop.ring, ring_pipeline_params);
    RingPipelineState ring_pipe_consumer_state;
    RingPipelineState ring_pipe_producer_state = cutlass::make_producer_start_state<RingPipeline>();
    typename CollectiveMainloop::PipelineState mainloop_pipe_transform_state;   // TW: the transform warps' consumer state on the TMA pipeline

    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Epilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    } 
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumEpilogueLoadThreads;
    epi_load_pipeline_params.consumer_arv_count = NumMMAThreads;
    if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
      epi_load_pipeline_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename LoadWarpOrderBarrier::Params params_load_order_barrier;
    params_load_order_barrier.group_id = producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
    params_load_order_barrier.group_size = NumThreadsPerWarp;
    LoadWarpOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, params_load_order_barrier);

    // Initialize starting pipeline states for the collectives
    // Epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    // For the DMA Load (producer) we start with an opposite phase
    // i.e., we skip all waits since we know that the buffer is indeed empty
    PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();


    auto cluster_wait_fn = [] () {
      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer thread blocks in the Cluster
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return [] () { cute::cluster_wait(); };
      }
      else {
        __syncthreads();
        return [] () {}; // do nothing
      }
    } ();

    // Optionally append 1s until problem shape is rank-4 in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    TiledMma tiled_mma;
    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)

    TileScheduler scheduler{params.scheduler};
    if constexpr (IsSchedDynamicPersistent) {
      scheduler.set_data_ptr(shared_storage.scheduler.data());
    }
    // Declare work_tile_info, then define it in each of warps that use it.
    typename TileScheduler::WorkTileInfo work_tile_info;

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop;

    // Prepare and partition the input tensors. Expects a tuple of tensors where:
    // get<0>(load_inputs) is the tma tensor A after local tiling so that it has shape (BLK_M,BLK_K,m,k,l)
    // get<1>(load_inputs) is the tma tensor B after local tiling so that it has shape (BLK_N,BLK_K,n,k,l)
    auto load_inputs = collective_mainloop.load_init(problem_shape_MNKL, params.mainloop);
    static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

    // Extract out partitioned A and B.
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    if (warp_group_role == WarpGroupRole::Producer) {
      work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
      if constexpr (LoadRegisterRequirement > 0) { cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>(); }   // TW
      if (is_transform_warp && is_epi_load_needed) { __trap(); }   // TW: warp 2 cannot be both the epilogue source-load warp and a transform warp

      // TW: transform warps
      if (is_transform_warp) {
        while (work_tile_info.is_valid()) {
          if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
            auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
            work_tile_info = next_work_tile_info;
            continue;
          }
          auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);
          collective_mainloop.transform(
            mainloop_pipeline,
            mainloop_pipe_transform_state,
            ring_pipeline,
            ring_pipe_producer_state,
            work_k_tile_count,
            transform_thread_idx,
            shared_storage.tensors.mainloop,
            shared_storage.pipelines.mainloop
          );
          mainloop_pipe_transform_state.advance(work_k_tile_count);
          ring_pipe_producer_state.advance(work_k_tile_count * CollectiveMainloop::RingSub);   // TW: one (or KBLOCKS) ring barrier(s) per k-tile

          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info,
                                                                            scheduler_pipeline,
                                                                            scheduler_pipe_consumer_state
                                                                           );
          work_tile_info = next_work_tile_info;
          if constexpr (IsSchedDynamicPersistent) {
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }
          }
        }
        collective_mainloop.transform_tail(ring_pipeline, ring_pipe_producer_state);
      }
      else
      // Scheduler Producer Warp
      if (producer_warp_role == ProducerWarpRole::Warp1) {
        if constexpr (IsSchedDynamicPersistent) { 
          bool requires_clc_query = true;
          TileSchedulerPipelineState scheduler_pipe_producer_state = cutlass::make_producer_start_state<TileSchedulerPipeline>();

          cutlass::arch::wait_on_dependent_grids();
          while (work_tile_info.is_valid()) {

            if (requires_clc_query) {
              // Throttle CLC query to mitigate workload imbalance caused by skews among persistent workers.
              scheduler_throttle_pipeline.consumer_wait(scheduler_pipe_throttle_consumer_state);
              scheduler_throttle_pipeline.consumer_release(scheduler_pipe_throttle_consumer_state);
              ++scheduler_pipe_throttle_consumer_state;

              // Query next work tile
              scheduler_pipe_producer_state = scheduler.advance_to_next_work(scheduler_pipeline, scheduler_pipe_producer_state);
            }

            // Fetch next work tile
            auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(
              work_tile_info,
              scheduler_pipeline,
              scheduler_pipe_consumer_state
            );
            requires_clc_query = increment_pipe;
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }

            work_tile_info = next_work_tile_info;
          }
          scheduler_pipeline.producer_tail(scheduler_pipe_producer_state);
        } 
      } // Scheduler Producer Warp End  
      else

      // Mainloop Producer Warp
      if (producer_warp_role == ProducerWarpRole::Mainloop) {
        // Ensure that the prefetched kernel does not touch
        // unflushed global memory prior to this instruction
        cutlass::arch::wait_on_dependent_grids();
        bool do_load_order_arrive = true;
        bool requires_clc_query = true;
        while (work_tile_info.is_valid()) {
          if (!TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
            auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info);
            work_tile_info = next_work_tile_info;   
            continue;
          }

          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

          // Get the number of K tiles to compute for this work as well as the starting K tile offset of the work.
          auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);
          auto work_k_tile_start = TileScheduler::get_work_k_tile_start(work_tile_info);
          auto k_tile_iter = cute::make_coord_iterator(idx2crd(work_k_tile_start, shape<3>(gA_mkl)), shape<3>(gA_mkl));

          if (requires_clc_query) {
            scheduler_throttle_pipeline.producer_acquire(scheduler_pipe_throttle_producer_state);
            scheduler_throttle_pipeline.producer_commit(scheduler_pipe_throttle_producer_state);
            ++scheduler_pipe_throttle_producer_state;
          }

          collective_mainloop.load(
            params.mainloop,
            mainloop_pipeline,
            mainloop_pipe_producer_state,
            load_inputs,
            blk_coord,
            k_tile_iter, work_k_tile_count,
            lane_idx,
            block_rank_in_cluster,
            shared_storage.tensors.mainloop
          );
          // Update starting pipeline state for the next tile
          mainloop_pipe_producer_state.advance(work_k_tile_count);

          // Signal for the epilogue load warp to begin
          if (do_load_order_arrive) {
            load_order_barrier.arrive();
            do_load_order_arrive = false;
          }
          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info,
                                                                            scheduler_pipeline,             
                                                                            scheduler_pipe_consumer_state
                                                                           );

          work_tile_info = next_work_tile_info;
          if constexpr (IsSchedDynamicPersistent) { 
            requires_clc_query = increment_pipe; 
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }
          }
        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

      }
      // TW: MainloopAux role removed (warp 3 is a transform warp)

      // Epilogue Producer Warp
      else if (producer_warp_role == ProducerWarpRole::Epilogue && is_epi_load_needed) {

        // Ensure that the prefetched kernel does not touch
        // unflushed global memory prior to this instruction
        cutlass::arch::wait_on_dependent_grids();

        if (!TileScheduler::requires_separate_reduction(params.scheduler) && work_tile_info.is_valid()) {
          load_order_barrier.wait();
        }

        CollectiveEpilogue collective_epilogue(params.epilogue, epi_tensors(0));

        while (work_tile_info.is_valid()) {
          if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
            // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
            auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
            auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
            
            epi_load_pipe_producer_state =
            collective_epilogue.load(
              epi_load_pipeline,
              epi_load_pipe_producer_state,
              problem_shape_MNKL,
              blk_shape,
              blk_coord,
              tiled_mma,
              lane_idx,
              epi_tensors(0),
              work_tile_info.reduction_subtile_idx()
            );
          }

          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info,
                                                                            scheduler_pipeline,     
                                                                            scheduler_pipe_consumer_state
                                                                           );
          work_tile_info = next_work_tile_info;
          if constexpr (IsSchedDynamicPersistent) { 
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }
          }
        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
      } // Epilogue Producer Warp End
    } // Producer Warp Group End

    else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});
      if constexpr (MmaRegisterRequirement > 0) { cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>(); }   // TW

      CollectiveEpilogue collective_epilogue(params.epilogue, epi_tensors(0));

      // Do we potentially issue tail arrives for TMA stores, if epilogue load is waiting for it
      bool do_store_tail = false;
      typename CollectiveMainloop::MmaTail mma_tail{};   // TW: the ring entry of the last k-tile (deferred release)
      [[maybe_unused]] bool mma_ran = false;
#if NVFP4_TW_TIMERS
      unsigned long long tl_body = 0, tl_epi = 0, tl_fetch = 0, tl_tiles = 0;   // TW: per-tile accounting
#endif
      while (work_tile_info.is_valid()) {
#if NVFP4_TW_TIMERS
        unsigned long long const tl_t0 = clock64();
#endif
        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);
        auto work_k_tile_count = TileScheduler::get_work_k_tile_count(work_tile_info, problem_shape_MNKL, blk_shape);
        // Allocate the accumulators for the (M,N) blk_shape
        //
        // MSVC CTAD breaks if we say "Tensor" here, so we use "auto" instead.
        auto accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));                 // (MMA,MMA_M,MMA_N)
        if (TileScheduler::valid_warpgroup_in_work_tile(work_tile_info)) {
          mma_tail = collective_mainloop.mma(     // TW: A from the TMA stage, B from the ring
            mainloop_pipeline,
            mainloop_pipe_consumer_state,
            ring_pipeline,
            ring_pipe_consumer_state,
            accumulators,
            work_k_tile_count,
            mma_thread_idx,
            shared_storage.tensors.mainloop,
            shared_storage.pipelines.mainloop
          );
          mma_ran = true;

          // Make sure the math instructions are done and free buffers before entering the epilogue
          collective_mainloop.mma_tail(
            mainloop_pipeline,
            mainloop_pipe_consumer_state,
            work_k_tile_count
          );

          // Update starting mainloop pipeline state for the next tile
          mainloop_pipe_consumer_state.advance(work_k_tile_count);
          ring_pipe_consumer_state.advance(work_k_tile_count * CollectiveMainloop::RingSub);   // TW: one (or KBLOCKS) ring barrier(s) per k-tile
        }

        if constexpr (!IsSm120Family) {
          if (scheduler.is_last_tile(work_tile_info)) {
            // Hint on an early release of global memory resources.
            // The timing of calling this function only influences performance,
            // not functional correctness.
            cutlass::arch::launch_dependent_grids();

          }
        }

        // Index of warp group within consumer warp groups
        int consumer_warp_group_idx = canonical_warp_group_idx() - NumLoadWarpGroups;

        // Perform reduction across splits, if needed
        TileScheduler::fixup(
          params.scheduler, work_tile_info, accumulators, NumMmaWarpGroups, consumer_warp_group_idx);

#if NVFP4_TW_TIMERS
        unsigned long long const tl_t1 = clock64();
#endif
        if (TileScheduler::compute_epilogue(work_tile_info, params.scheduler)) {
          // Epilogue and write to gD
          auto [epi_load_pipe_consumer_state_next, epi_store_pipe_producer_state_next] =
          collective_epilogue.store(
            epi_load_pipeline,
            epi_load_pipe_consumer_state,
            epi_store_pipeline,
            epi_store_pipe_producer_state,
            problem_shape_MNKL,
            blk_shape,
            blk_coord,
            accumulators,
            tiled_mma,
            mma_thread_idx,
            epi_tensors(mma_tail.last_entry),
            work_tile_info.reduction_subtile_idx()
          );
          epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next;
          epi_store_pipe_producer_state = epi_store_pipe_producer_state_next;
          do_store_tail = true;
        }
#if NVFP4_TW_RING_EPI
        // TW: the epilogue borrowed ring entry mma_tail.last_entry. Drain this tile's TMA stores (bulk wait_group.read 0
        // by the issuing warp), then every MMA warp arrives on the entry's empty barrier(s) so the transform warps may
        // fill it with the next tile's k-tile 1 (k-tile 0 is already in the other entry).
        if (mma_ran) {
          auto [epi_load_tail_state, epi_store_tail_state] = collective_epilogue.store_tail(
            epi_load_pipeline, epi_load_pipe_consumer_state, epi_store_pipeline, epi_store_pipe_producer_state);
          epi_load_pipe_consumer_state = epi_load_tail_state;
          epi_store_pipe_producer_state = epi_store_tail_state;
          if constexpr (CollectiveMainloop::kDep) {
            auto rel = mma_tail.last_rel;
            CUTLASS_PRAGMA_UNROLL
            for (int s = 0; s < CollectiveMainloop::RingSub; ++s) {
#if NVFP4_TW_WARP_ARRIVE
              cutlass::gemm::collective::nvfp4_tw_detail::warp_arrive32(shared_storage.pipelines.mainloop.ring.empty_barrier_[rel.index()]);
#else
              ring_pipeline.consumer_release(rel);
#endif
              ++rel;
            }
          }
          mma_ran = false;
        }
#endif
#if NVFP4_TW_TIMERS
        unsigned long long const tl_t2 = clock64();
#endif

        // Get next work tile
        auto [next_work_tile_info, increment_pipe] = scheduler.fetch_next_work(work_tile_info,
                                                                          scheduler_pipeline,
                                                                          scheduler_pipe_consumer_state
                                                                          );
        work_tile_info = next_work_tile_info;
        if constexpr (IsSchedDynamicPersistent) { 
          if (increment_pipe) {
            ++scheduler_pipe_consumer_state;
          }
        }
#if NVFP4_TW_TIMERS
        {
          unsigned long long const tl_t3 = clock64();
          tl_body += tl_t3 - tl_t0; tl_epi += tl_t2 - tl_t1; tl_fetch += tl_t3 - tl_t2; ++tl_tiles;
        }
#endif
      } // Scheduler work fetch loop
#if NVFP4_TW_TIMERS
      if (lane_idx == 0) {
        using cutlass::gemm::collective::nvfp4_tw_detail::g_tw_timers;
        atomicAdd(&g_tw_timers[8], tl_body); atomicAdd(&g_tw_timers[9], tl_tiles);
        atomicAdd(&g_tw_timers[10], tl_epi); atomicAdd(&g_tw_timers[11], tl_fetch);
      }
#endif

      if (do_store_tail) {
        collective_epilogue.store_tail(
          epi_load_pipeline,
          epi_load_pipe_consumer_state,
          epi_store_pipeline,
          epi_store_pipe_producer_state
        );
      }
    } // Consumer Warp Groups End
#endif
  }

};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
