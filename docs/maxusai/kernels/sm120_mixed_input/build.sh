#!/usr/bin/env bash
# Compile one kernel source into a self-contained shared library, inside a container that has nvcc 12.8
# (no CUDA toolchain on the host). No GPU is needed to compile. Usage: ./build.sh src.cu out.so [extra nvcc flags]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CUTLASS=${CUTLASS:-/mnt/8TB_SN850X_RAID1_BTRFS/claude-scratch/cutlass}
IMG=${IMG:-nvcr.io/nvidia/tritonserver:25.02-py3}   # nvcc 12.8.61
SRC=$1; OUT=$2; shift 2
exec docker run --rm -u "$(id -u):$(id -g)" -v "$CUTLASS":/cutlass:ro -v "$HERE":/work -w /work "$IMG" \
  nvcc -std=c++17 -O3 -arch=sm_120a --expt-relaxed-constexpr -DNDEBUG \
       -I/cutlass/include -I/cutlass/tools/util/include \
       -Xcompiler -fPIC -shared -cudart static "$@" -o "$OUT" "$SRC"
