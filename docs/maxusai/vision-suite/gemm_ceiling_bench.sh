#!/bin/sh
# Build and run gemm_ceiling_bench.cpp: what fraction of the hardware ceiling
# does rocBLAS actually reach on the shapes this fork's vision models generate?
#
# Every other measurement in this repo is COMPARATIVE -- it says which config is
# faster, never whether the fastest one is any good. This one has an absolute
# denominator, which is also why it needs SPEC H24: a ratio against a ceiling is
# only as good as the ceiling.
#
# Needs a ROCm image with hipcc and rocBLAS; the slim runtime image has neither.
# LD_LIBRARY_PATH is required at RUN time -- librocblas lives in /opt/rocm/lib
# and that image leaves LD_LIBRARY_PATH empty.
set -eu
IMAGE=${IMAGE:-rocm/dev-ubuntu-24.04:10.0.0-full}
ARCH=${ARCH:-gfx1151}
HERE=$(cd "$(dirname "$0")" && pwd)
docker run --rm -v "$HERE:/w" -w /w "$IMAGE" \
  hipcc --offload-arch="$ARCH" -O3 gemm_ceiling_bench.cpp -o gemm_ceiling_bench -lrocblas
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  -e LD_LIBRARY_PATH=/opt/rocm/lib -v "$HERE:/w" -w /w "$IMAGE" ./gemm_ceiling_bench
