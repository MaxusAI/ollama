#!/usr/bin/env bash
# Measure the one gap llama.cpp#29953's current head still has: the native-FP4 y scales.
#
# The RTX PRO 6000 Blackwell compiles as sm_120a, so this host can take the native-FP4 path after all --
# blackwell_mma_available() only needs highest_compiled_arch >= 1200, and test-backend-ops already has
# MUL_MAT_ID_W4A4 cases that reach GGML_PREC_Q4. So the y-scale read can be measured here rather than argued
# from the source.
#
# Two builds of the PR's head (3070d927f), differing only in src1_scale's size:
#   head        as published: ids_dst and src1 padded, src1_scale not
#   headyscale  + src1_scale.alloc(... + J_best-1), i.e. tasks/mmq-amend-29953-yscale.patch
# and four allocator modes, with the guard page on one buffer at a time so a fault is attributable.
set -u
cd "$(dirname "$0")"
HERE=$PWD
SRC=wt-29953head
OUT=$HERE/mmq-nvfp4-results
BUILD=build-mmq-120a
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
GPU=${GPU:-0}
JOBS=${JOBS:-8}
REPS=${REPS:-3}
MIN_FREE_MIB=${MIN_FREE_MIB:-20480}
declare -A CASES=(
    # 40 tiles at 21% of the waves on 188 SMs, so launch_mul_mat_q uses nsm blocks and runs the fixup kernel
    [fixup]='type_a=nvfp4,type_b=f32,n_mats=8,n_used=1,b=0,m=640,n=9,k=512,'
    # 2560 tiles at 97%, so blocks == ntiles, fixup_needed is false and nothing reads y_scale unbounded
    [nofixup]='type_a=nvfp4,type_b=f32,n_mats=256,n_used=16,b=0,m=640,n=16,k=512,'
)
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
mkdir -p "$OUT/bin" "$OUT/logs"
TSV=$OUT/results.tsv
[ -s "$TSV" ] || printf 'case\tvariant\tmode\trep\tpassed\tillegal_access\n' > "$TSV"

cd "$SRC" || exit 1
trap 'git checkout --quiet -- . 2>/dev/null' EXIT
git checkout --quiet -- .
grep -q '8, 1, false, 640, 9, 512' tests/test-backend-ops.cpp \
    || git apply "$HERE/mmq-nvfp4-test.patch" || { echo "the nvfp4 test patch does not apply"; exit 1; }

cmake -S . -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="120a" -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
      -DCUDAToolkit_ROOT="$CUDA_HOME" -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=ON \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF \
      > "$OUT/logs/cmake.log" 2>&1 || { echo "cmake failed, see $OUT/logs/cmake.log"; exit 1; }

for v in head headyscale; do
    [ -x "$OUT/bin/test-backend-ops-$v" ] && continue
    git checkout --quiet -- ggml/src/ggml-cuda/mmq.cu ggml/src/ggml-cuda/mmq.cuh
    python3 "$HERE/mmq-variant.py" . "$v" --debug > /dev/null || exit 1
    echo "building $v ..."
    ninja -C "$BUILD" -j "$JOBS" test-backend-ops > "$OUT/logs/build-$v.log" 2>&1 \
        || { echo "build $v failed, see $OUT/logs/build-$v.log"; exit 1; }
    cp "$BUILD"/bin/test-backend-ops "$OUT/bin/test-backend-ops-$v"
done
git checkout --quiet -- ggml/src/ggml-cuda/mmq.cu ggml/src/ggml-cuda/mmq.cuh

free_mib() { nvidia-smi --id="$GPU" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
for c in fixup nofixup; do for v in head headyscale; do for mode in off guard:src1_scale guard:src1 guard:ids_dst; do
    for r in $(seq 1 "$REPS"); do
        log=$OUT/logs/${c}_${v}_${mode//:/_}_r$r.log
        grep -q '^RESULT' "$log" 2>/dev/null && continue
        while [ "$(free_mib)" -lt "$MIN_FREE_MIB" ]; do sleep 60; done
        env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" MMQ_DEBUG_ALLOC="$mode" MMQ_DEBUG_PRINT=1 \
            timeout 900 "$OUT/bin/test-backend-ops-$v" test -o MUL_MAT_ID_W4A4 -b CUDA0 -p "${CASES[$c]}" > "$log" 2>&1
        pass=$(grep -qE '^ *1/1 tests passed' "$log" && echo 1 || echo 0)
        ill=$(grep -c 'illegal memory access' "$log")
        printf 'RESULT\t%s\t%s\t%s\t%s\n' "$v" "$mode" "$r" "$pass" >> "$log"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$c" "$v" "$mode" "$r" "$pass" "$ill" >> "$TSV"
        printf '%-8s %-12s %-18s r%s  pass=%s illegal=%s\n' "$c" "$v" "$mode" "$r" "$pass" "$ill"
    done
done; done; done

echo "NVFP4 COMPLETE $(date -u +%FT%TZ)"
awk -F'\t' 'NR>1 {k=$1" "$2" "$3; n[k]++; if ($5=="1") p[k]++}
            END {for (k in n) printf "  %-40s %d/%d passed\n", k, p[k]+0, n[k]}' "$TSV" | sort
grep -hE "MMQ_IDS|MMQ_SCALE" "$OUT"/logs/fixup_head_off_r1.log 2>/dev/null
