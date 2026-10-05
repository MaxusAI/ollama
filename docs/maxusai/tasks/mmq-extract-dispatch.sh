#!/usr/bin/env bash
# Cut ggml_cuda_should_use_mmq() and the MUL_MAT_ID MMVQ batch limits out of a llama.cpp checkout, verbatim,
# so a host-side check compiles the dispatch the build actually uses instead of a re-typed copy.
#   mmq-extract-dispatch.sh <llama.cpp checkout> > dispatch.inc
set -eu
C=${1:?usage: $0 <llama.cpp checkout>}
cuda=$C/ggml/src/ggml-cuda
printf '// cut from %s at %s -- do not edit\n' "$C" "$(git -C "$C" rev-parse --short HEAD)"
# get_mmvq_mmid_max_batch_*() and get_mmvq_mmid_max_batch(): from the first helper to the end of the host function
awk '/^static constexpr __host__ __device__ int get_mmvq_mmid_max_batch_/ {on=1}
     on {print}
     on && /^int get_mmvq_mmid_max_batch\(/ {host=1}
     host && /^}/ {exit}' "$cuda/mmvq.cu"
# ggml_cuda_should_use_mmq(): signature line to the closing brace in column 0
awk '/^bool ggml_cuda_should_use_mmq\(/ {on=1} on {print} on && /^}/ {exit}' "$cuda/mmq.cu"
