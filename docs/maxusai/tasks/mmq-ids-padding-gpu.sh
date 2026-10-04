#!/usr/bin/env bash
# GPU check for llama.cpp#27044 against #29941: does MUL_MAT_ID's MMQ path read past the src1 buffer?
# See docs/maxusai/upstream-mmq-submission-material.md, "Results on sm_120 (2026-10-04)".
#
#   mmq-ids-padding-gpu.sh <llama.cpp checkout> [out dir]
#
# The checkout should be at master 05043961, or close to it. The script:
#   1. applies mmq-ids-padding-gpu.patch: six test_mul_mat_id cases, the MMQ445_EXACT debug switch, and the
#      mm_ids_helper launch workaround (all common to every variant);
#   2. builds test-backend-ops three times, changing only the argument of the ids-path padding line:
#      master (ne11), #29941 (ne12) and #27044 (ne12*n_expert_used);
#   3. runs every case under compute-sanitizer memcheck, one process per case, in two modes:
#      exact (MMQ445_EXACT=1: the src1 buffer gets its own exact-size cudaMalloc) and stock (the memory pool).
#
# Environment: CUDA_HOME (default /usr/local/cuda), CUDA_ARCH (default 120a), GPU (default 0, by PCI bus order),
# JOBS (default 12), MIN_FREE_MIB (default 4096: free memory the GPU must have before each run).
set -u
SRC=${1:?usage: $0 <llama.cpp checkout> [out dir]}
OUT=${2:-$PWD/mmq-ids-padding-results}
HERE=$(cd "$(dirname "$0")" && pwd)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
SAN=$CUDA_HOME/bin/compute-sanitizer
CUDA_ARCH=${CUDA_ARCH:-120a}
GPU=${GPU:-0}
JOBS=${JOBS:-12}
MIN_FREE_MIB=${MIN_FREE_MIB:-4096}
mkdir -p "$OUT/bin" "$OUT/logs"

cleanup() { pkill -P $$ 2>/dev/null; }
trap cleanup EXIT

cd "$SRC" || exit 1
if ! grep -q MMQ445_EXACT ggml/src/ggml-cuda/mmq.cu; then
    git apply "$HERE/mmq-ids-padding-gpu.patch" || { echo "patch does not apply to this checkout"; exit 1; }
fi
cmake -S . -B build-mmq445 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
      -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=ON \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF > "$OUT/logs/cmake.log" 2>&1 || exit 1

# The ids-path padding line, as master has it. Each variant changes only its last argument.
F=ggml/src/ggml-cuda/mmq.cu
PAD='ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) \* sizeof(block_q8_1_mmq);'
[ "$(grep -c "$PAD" $F)" = 2 ] || { echo "expected the padding line twice in $F (dense and ids paths)"; exit 1; }
cp $F "$OUT/mmq.cu.master"
for v in master 29941 ours; do
    cp "$OUT/mmq.cu.master" $F
    case $v in
        29941) arg='ne12' ;;
        ours)  arg='ne12*n_expert_used' ;;
        *)     arg='' ;;
    esac
    # the second occurrence is the ids path; the first, the dense path, stays as it is
    [ -n "$arg" ] && awk -v a="$arg" '/ggml_cuda_mmq_get_J_max\(src0->type, fallback, cc, ne11\) \* sizeof\(block_q8_1_mmq\);/ { n++; if (n == 2) sub(/cc, ne11\)/, "cc, " a ")") } { print }' \
        "$OUT/mmq.cu.master" > $F
    ninja -C build-mmq445 -j "$JOBS" test-backend-ops > "$OUT/logs/build-$v.log" 2>&1 || { cp "$OUT/mmq.cu.master" $F; exit 1; }
    cp build-mmq445/bin/test-backend-ops "$OUT/bin/test-backend-ops-$v"
done
cp "$OUT/mmq.cu.master" $F

declare -A CASES=(
    [c4_fb_65]='type_a=q4_K,type_b=f32,n_mats=256,n_used=8,b=1,m=576,n=65,k=2048,'
    [c5_100]='type_a=q4_K,type_b=f32,n_mats=256,n_used=8,b=1,m=512,n=100,k=2048,'
    [c6_29847_100]='type_a=q4_0,type_b=f32,n_mats=512,n_used=10,b=1,m=640,n=100,k=2560,'
    [c3_orig_2040]='type_a=q4_K,type_b=f32,n_mats=256,n_used=8,b=1,m=512,n=2040,k=2048,'
    [c1_29847_b0]='type_a=q4_0,type_b=f32,n_mats=512,n_used=10,b=0,m=640,n=508,k=2560,'
    [c2_29847_b1]='type_a=q4_0,type_b=f32,n_mats=512,n_used=10,b=1,m=640,n=508,k=2560,'
)
free_mib() { nvidia-smi --id="$GPU" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
TSV=$OUT/results.tsv
printf 'case\tvariant\tmode\tmemcheck_errors\ttest_passed\n' > "$TSV"
for c in c4_fb_65 c5_100 c6_29847_100 c3_orig_2040 c1_29847_b0 c2_29847_b1; do
    for mode in exact stock; do
        for v in master 29941 ours; do
            while [ "$(free_mib)" -lt "$MIN_FREE_MIB" ]; do sleep 60; done
            log=$OUT/logs/${c}_${v}_${mode}.log
            envx=(); [ $mode = exact ] && envx=(MMQ445_EXACT=1)
            env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" "${envx[@]}" timeout 1800 \
                "$SAN" --tool memcheck --show-backtrace no --print-limit 3 \
                "$OUT/bin/test-backend-ops-$v" test -o MUL_MAT_ID -b CUDA0 -p "${CASES[$c]}" > "$log" 2>&1
            errs=$(grep -oE 'ERROR SUMMARY: [0-9]+ errors?$' "$log" | grep -oE '[0-9]+' | tail -1)
            if grep -qE '^ *1/1 tests passed' "$log"; then passed=yes; else passed=no; fi
            printf '%s\t%s\t%s\t%s\t%s\n' "$c" "$v" "$mode" "${errs:-?}" "$passed" | tee -a "$TSV"
        done
    done
done
