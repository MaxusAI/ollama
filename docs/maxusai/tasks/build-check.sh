#!/usr/bin/env bash
# Build a GPU-free MMQ check against a llama.cpp checkout.
#
#   build-check.sh <llama.cpp checkout> <source.cu> <output>
#
# The gencode list is load-bearing: ggml_cuda_highest_compiled_arch() reads __CUDA_ARCH_LIST__, and
# ggml_cuda_mmq_get_config() picks a different config table when the arch it is asked about was not compiled for.
# Without these flags every NVIDIA arch resolves to nvcc's default and the configs come out wrong (J capped at 64
# on sm_120 instead of 128), so the whole check silently models a machine that does not exist.
set -eu
SRC=$(cd "${1:?usage: $0 <llama.cpp checkout> <source.cu> <output>}" && pwd)
CU=$(cd "$(dirname "${2:?}")" && pwd)/$(basename "$2")
OUT=$3
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
cd "$SRC"
"$CUDA_HOME/bin/nvcc" -Wno-deprecated-gpu-targets -std=c++17 \
    -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 \
    -gencode arch=compute_86,code=sm_86 \
    -gencode arch=compute_89,code=sm_89 \
    -gencode arch=compute_120,code=sm_120 \
    -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cuda "$CU" -o "$OUT" -lcublas -lcuda
