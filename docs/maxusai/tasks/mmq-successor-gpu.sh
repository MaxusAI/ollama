#!/usr/bin/env bash
# GPU check for a successor to llama.cpp#27044, on master with #29941 merged (dd266785c): which padding rule keeps
# MUL_MAT_ID's MMQ path inside its buffers? See docs/maxusai/upstream-mmq-successor-material.md.
#
#   mmq-successor-gpu.sh <llama.cpp checkout> [out dir]
#
# The checkout should be at dd266785c or later, with ggml_cuda_mul_mat_q's ids branch still padding
# get_J_max(ne12). The script:
#   1. applies mmq-successor-tests.patch (ten test_mul_mat_id cases) unless the checkout already has them;
#   2. builds test-backend-ops eight times, changing only ggml/src/ggml-cuda/mmq.cu (mmq-successor-variant.py):
#      four padding rules (pre29941 = ne11, master = ne12, 27044 = ne12*n_expert_used, successor = the widest
#      tile, ids_dst included), each without and with the MMQ_EXACT debug switch;
#   3. runs every case, one process per case, three ways:
#        stock    the binary without the switch, no sanitizer, REPS times: does the test pass or abort?
#        memcheck the same binary under compute-sanitizer memcheck: the memory pool as everyone runs it
#        exact    the switch binary with MMQ_EXACT=1 under memcheck: ids_dst and src1_q8_1 in exact-size
#                 allocations, so a read past either is an error even when the pool would have hidden it.
#
# Environment: CUDA_HOME (default /usr/local/cuda), CUDA_ARCH (default 120), GPU (default 0, by PCI bus order),
# JOBS (default 12), REPS (default 5), MIN_FREE_MIB (default 4096: free memory the GPU must have before each run),
# MODES (default "stock memcheck exact").
set -u
SRC=${1:?usage: $0 <llama.cpp checkout> [out dir]}
OUT=${2:-$PWD/mmq-successor-results}
HERE=$(cd "$(dirname "$0")" && pwd)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
SAN=$CUDA_HOME/bin/compute-sanitizer
CUDA_ARCH=${CUDA_ARCH:-120}
GPU=${GPU:-0}
JOBS=${JOBS:-12}
REPS=${REPS:-5}
MIN_FREE_MIB=${MIN_FREE_MIB:-4096}
MODES=${MODES:-stock memcheck exact}
VARIANTS="pre29941 master 27044 successor"
mkdir -p "$OUT/bin" "$OUT/logs"
F=ggml/src/ggml-cuda/mmq.cu
# CMake hands nvcc the toolkit's include directory. A search path from a shell profile (CPATH and friends) is searched
# before it, and one naming another toolkit mixes headers: CUDA 12.1's crt/host_config.h under nvcc 12.8 refuses gcc 13.
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH

cd "$SRC" || exit 1
restore() { [ -f "$OUT/mmq.cu.master" ] && cp "$OUT/mmq.cu.master" "$F"; pkill -P $$ 2>/dev/null; }
trap restore EXIT
grep -q 'ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne12) \* sizeof(block_q8_1_mmq);' "$F" \
    || { echo "$F does not pad the ids branch with get_J_max(ne12): this is not master with #29941"; exit 1; }
if ! grep -q 'get_J_max(113) = 64' tests/test-backend-ops.cpp; then
    git apply "$HERE/mmq-successor-tests.patch" || { echo "mmq-successor-tests.patch does not apply to this checkout"; exit 1; }
fi
cmake -S . -B build-mmq-successor -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
      -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME" -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=ON \
      -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF > "$OUT/logs/cmake.log" 2>&1 || exit 1

cp "$F" "$OUT/mmq.cu.master"
for v in $VARIANTS; do for e in stock exact; do
    [ -x "$OUT/bin/test-backend-ops-$v-$e" ] && continue
    cp "$OUT/mmq.cu.master" "$F"
    python3 "$HERE/mmq-successor-variant.py" "$F" "$v" $([ $e = exact ] && echo --exact) > /dev/null || exit 1
    ninja -C build-mmq-successor -j "$JOBS" test-backend-ops > "$OUT/logs/build-$v-$e.log" 2>&1 || exit 1
    cp build-mmq-successor/bin/test-backend-ops "$OUT/bin/test-backend-ops-$v-$e"
done; done
cp "$OUT/mmq.cu.master" "$F"

# The ten cases of mmq-successor-tests.patch, in its order.
CASES="p29847_b0 p29847_b1 j100_b0 j100_b1 e120_b0 e120_b1 one113 orig2040 fb65 t100"
declare -A P=(
    [p29847_b0]='type_a=q4_0,type_b=f32,n_mats=512,n_used=10,b=0,m=640,n=508,k=2560,'
    [p29847_b1]='type_a=q4_0,type_b=f32,n_mats=512,n_used=10,b=1,m=640,n=508,k=2560,'
    [j100_b0]='type_a=q4_0,type_b=f32,n_mats=512,n_used=10,b=0,m=640,n=100,k=2560,'
    [j100_b1]='type_a=q4_0,type_b=f32,n_mats=512,n_used=10,b=1,m=640,n=100,k=2560,'
    [e120_b0]='type_a=q4_K,type_b=f32,n_mats=256,n_used=10,b=0,m=576,n=120,k=1536,'
    [e120_b1]='type_a=q4_K,type_b=f32,n_mats=256,n_used=10,b=1,m=576,n=120,k=1536,'
    [one113]='type_a=q4_K,type_b=f32,n_mats=64,n_used=1,b=0,m=576,n=113,k=16384,'
    [orig2040]='type_a=q4_K,type_b=f32,n_mats=256,n_used=8,b=1,m=512,n=2040,k=2048,'
    [fb65]='type_a=q4_K,type_b=f32,n_mats=256,n_used=8,b=1,m=576,n=65,k=2048,'
    [t100]='type_a=q4_K,type_b=f32,n_mats=256,n_used=8,b=1,m=512,n=100,k=2048,'
)
free_mib() { nvidia-smi --id="$GPU" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
run() { # <log> <binary> <env...> -- <pattern> [sanitizer]
    local log=$1 bin=$2 pat san; shift 2; local envx=()
    while [ "$1" != -- ]; do envx+=("$1"); shift; done; shift; pat=$1 san=${2:-}
    while [ "$(free_mib)" -lt "$MIN_FREE_MIB" ]; do sleep 60; done
    if [ -n "$san" ]; then
        env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" "${envx[@]}" timeout 1800 \
            "$SAN" --tool memcheck --show-backtrace no --print-limit 3 "$bin" test -o MUL_MAT_ID -b CUDA0 -p "$pat" > "$log" 2>&1
    else
        env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" "${envx[@]}" timeout 1800 \
            "$bin" test -o MUL_MAT_ID -b CUDA0 -p "$pat" > "$log" 2>&1
    fi
}
passed() { grep -qE '^ *1/1 tests passed' "$1" && echo 1 || echo 0; }
TSV=$OUT/results.tsv
[ -s "$TSV" ] || printf 'case\tvariant\tmode\trep\tpassed\tillegal_access\tmemcheck_errors\n' > "$TSV"
for c in $CASES; do for v in $VARIANTS; do
    for mode in $MODES; do
        case $mode in
            stock)    for r in $(seq 1 "$REPS"); do log=$OUT/logs/${c}_${v}_stock_r$r.log
                          run "$log" "$OUT/bin/test-backend-ops-$v-stock" -- "${P[$c]}"
                          ill=$(grep -c 'illegal memory access' "$log")
                          printf '%s\t%s\tstock\t%s\t%s\t%s\t\n' "$c" "$v" "$r" "$(passed "$log")" "$ill" >> "$TSV"; done ;;
            memcheck) log=$OUT/logs/${c}_${v}_memcheck.log
                      run "$log" "$OUT/bin/test-backend-ops-$v-stock" -- "${P[$c]}" san
                      errs=$(grep -oE 'ERROR SUMMARY: [0-9]+ errors?$' "$log" | grep -oE '[0-9]+' | tail -1)
                      printf '%s\t%s\tmemcheck\t1\t%s\t\t%s\n' "$c" "$v" "$(passed "$log")" "${errs:-?}" >> "$TSV" ;;
            exact)    log=$OUT/logs/${c}_${v}_exact.log
                      run "$log" "$OUT/bin/test-backend-ops-$v-exact" MMQ_EXACT=1 -- "${P[$c]}" san
                      errs=$(grep -oE 'ERROR SUMMARY: [0-9]+ errors?$' "$log" | grep -oE '[0-9]+' | tail -1)
                      printf '%s\t%s\texact\t1\t%s\t\t%s\n' "$c" "$v" "$(passed "$log")" "${errs:-?}" >> "$TSV" ;;
        esac
    done
done; done

# One line per case and variant: stock passes of REPS, memcheck errors with the pool, memcheck errors exact.
python3 - "$TSV" <<'EOF'
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
order = list(dict.fromkeys(r["case"] for r in rows)); variants = list(dict.fromkeys(r["variant"] for r in rows))
cell = collections.defaultdict(dict)
for r in rows:
    d = cell[(r["case"], r["variant"])]
    if r["mode"] == "stock":
        d.setdefault("stock", []).append(r["passed"] == "1")
    else:
        d[r["mode"]] = (r["memcheck_errors"], r["passed"] == "1")
def fmt(d):
    out = []
    if "stock" in d: out.append(f"{sum(d['stock'])}/{len(d['stock'])} pass")
    for m in ("memcheck", "exact"):
        if m in d: out.append(f"{m} {d[m][0]}{'' if d[m][1] else ' (aborted)'}")
    return "; ".join(out)
print("| case | " + " | ".join(variants) + " |")
print("|---" * (len(variants) + 1) + "|")
for c in order:
    print(f"| {c} | " + " | ".join(fmt(cell[(c, v)]) for v in variants) + " |")
EOF
