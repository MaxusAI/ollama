#!/usr/bin/env bash
# llama.cpp#29953 at its head (3070d927f), the exact tree, through the same harness as matrix.md and matrix-sm75.md.
# The published #29953 (p29953.patch, 5bd8b0013) runs alongside as the positive control: a pass in which the
# control does not fault on ids16 proves nothing about the head.
#
#   sm_120, GPU0  A: ids16, j100_b1, j100_b0 under guard:src1, 10 reps each, control and head
#                 B: every case, head only: stock, guard (both buffers), guard:src1, guard:ids_dst x3; exact x1
#   sm_75,  GPU1  C: ids16 and dense321, guard:src1 and guard:ids_dst x3, control
#                 D: ids16 and dense321, stock, guard:src1 and guard:ids_dst x3, head
#
# Runs from its own directory, as the other mmq-rules-* scripts here: llama.cpp-e2e next to it is a dd266785c
# checkout with mmq-tests.patch applied, and mmq-variant.py's head29953 swaps in the head's mmq.cu/mmq.cuh.
# Sequential on purpose: every build rewrites mmq.cu/mmq.cuh in the one checkout, so two passes may not overlap.
set -u
cd "$(dirname "$0")" || exit 1
HERE=$PWD
SRC=$HERE/llama.cpp-e2e
trap 'pkill -P $$ 2>/dev/null; git -C "$SRC" checkout --quiet -- ggml/src/ggml-cuda/mmq.cu ggml/src/ggml-cuda/mmq.cuh 2>/dev/null' EXIT

stamp() { echo "=== $(date -u +%FT%TZ) $*"; }

O120=$HERE/mmq-head-results-sm120
stamp "A: sm_120 control + head, guard:src1 x10"
BUILD_KINDS=debug VARIANTS="p29953 head29953" CASES="ids16 j100_b1 j100_b0" MODES=guard_src1 REPS=10 \
    CUDA_ARCH=120 GPU=0 BUILD_DIR=build-mmq-successor ./mmq-rules-gpu.sh "$SRC" "$O120" || { stamp "A FAILED"; exit 1; }

stamp "B: sm_120 head, every case"
BUILD_KINDS="stock debug" VARIANTS=head29953 MODES="stock guard guard_src1 guard_ids_dst exact" REPS=3 \
    CUDA_ARCH=120 GPU=0 BUILD_DIR=build-mmq-successor ./mmq-rules-gpu.sh "$SRC" "$O120" || { stamp "B FAILED"; exit 1; }

O75=$HERE/mmq-head-results-sm75
stamp "C: sm_75 control"
BUILD_KINDS=debug VARIANTS=p29953 CASES="ids16 dense321" MODES="guard_src1 guard_ids_dst" REPS=3 \
    CUDA_ARCH=75 GPU=1 MIN_FREE_MIB=4096 BUILD_DIR=build-mmq-sm75 ./mmq-rules-gpu.sh "$SRC" "$O75" || { stamp "C FAILED"; exit 1; }

stamp "D: sm_75 head"
BUILD_KINDS="stock debug" VARIANTS=head29953 CASES="ids16 dense321" MODES="stock guard_src1 guard_ids_dst" REPS=3 \
    CUDA_ARCH=75 GPU=1 MIN_FREE_MIB=4096 BUILD_DIR=build-mmq-sm75 ./mmq-rules-gpu.sh "$SRC" "$O75" || { stamp "D FAILED"; exit 1; }

stamp "leftover GPU processes from this run (expect none):"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -E 'test-backend-ops|compute-sanitizer' || echo "none"
stamp "HEAD29953 COMPLETE"
