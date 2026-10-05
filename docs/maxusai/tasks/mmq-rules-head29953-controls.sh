#!/usr/bin/env bash
# Positive controls for mmq-rules-head29953.sh's sm_120 pass B, in the same harness and output directory.
# Pass A controlled guard:src1 only. Pass B also ran the head under guard (both buffers), guard:ids_dst and exact
# (memcheck with exact-size allocations); a clean head there means something only if the same harness shows the
# published #29953 failing in those modes. ids16 is deterministic for both buffers; one113 is the case where the
# published rule gave the most memcheck errors in matrix.md (11,939).
#
# Runs from its own directory, after mmq-rules-head29953.sh (it reuses that pass's p29953-debug binary).
set -u
cd "$(dirname "$0")" || exit 1
HERE=$PWD
SRC=$HERE/llama.cpp-e2e
trap 'pkill -P $$ 2>/dev/null; git -C "$SRC" checkout --quiet -- ggml/src/ggml-cuda/mmq.cu ggml/src/ggml-cuda/mmq.cuh 2>/dev/null' EXIT
echo "=== $(date -u +%FT%TZ) E: sm_120 controls, published #29953"
BUILD_KINDS=debug VARIANTS=p29953 CASES="ids16 one113" MODES="guard guard_ids_dst exact" REPS=3 \
    CUDA_ARCH=120 GPU=0 BUILD_DIR=build-mmq-successor ./mmq-rules-gpu.sh "$SRC" "$HERE/mmq-head-results-sm120" \
    || { echo "E FAILED"; exit 1; }
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -E 'test-backend-ops|compute-sanitizer' || echo "no leftover GPU processes"
echo "=== $(date -u +%FT%TZ) CONTROLS COMPLETE"
