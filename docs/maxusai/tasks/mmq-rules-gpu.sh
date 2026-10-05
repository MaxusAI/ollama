#!/usr/bin/env bash
# Which MMQ MUL_MAT_ID padding rule keeps the kernel inside its buffers? Five rules, one checkout, one GPU.
#
#   mmq-rules-gpu.sh <llama.cpp checkout at dd266785c> [out dir]
#
# Rules (mmq-variant.py): ne11 (#24127, pre-#29941) | ne12 (#29941, master) | p29953 (#29953, the launched tile)
#                         | p27044 (#27044, our compat 903 until now) | successor (#448, the widest tile)
#
# Each rule is built twice, with and without the MMQ_DEBUG_ALLOC switch (mmq-debug-alloc.cuh), and every case is
# run in these modes:
#   stock         no switch, no sanitizer, REPS times -- what a user would hit
#   memcheck      no switch, under compute-sanitizer memcheck -- the memory pool as everyone runs it
#   exact         MMQ_DEBUG_ALLOC=exact under memcheck -- each buffer in its own exact-size cudaMalloc
#   guard         MMQ_DEBUG_ALLOC=guard -- each buffer at the top of its own mapping, next granule unmapped:
#                 a read past the end faults, no sanitizer needed
#   guard_ids_dst MMQ_DEBUG_ALLOC=guard:ids_dst -- only ids_dst guarded, so a fault is attributable to it
#   guard_src1    MMQ_DEBUG_ALLOC=guard:src1   -- only src1_q8_1 guarded
#
# Environment: CUDA_HOME (default /usr/local/cuda-12.8), CUDA_ARCH (default 120), GPU (default 0, PCI bus order),
# JOBS (default 8), REPS (default 3), MIN_FREE_MIB (default 20480: GPU0 is shared, leave the reserve alone),
# MODES, CASES, VARIANTS.
set -u
SRC=${1:?usage: $0 <llama.cpp checkout> [out dir]}
OUT=${2:-$PWD/mmq-rules-results}
HERE=$(cd "$(dirname "$0")" && pwd)
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
SAN=$CUDA_HOME/bin/compute-sanitizer
CUDA_ARCH=${CUDA_ARCH:-120}
GPU=${GPU:-0}
JOBS=${JOBS:-8}
REPS=${REPS:-3}
MIN_FREE_MIB=${MIN_FREE_MIB:-20480}
MODES=${MODES:-stock guard guard_ids_dst guard_src1 memcheck exact}
# One build directory per CUDA_ARCH, or the two architectures clobber each other's objects.
BUILD_DIR=${BUILD_DIR:-build-mmq-successor}
VARIANTS=${VARIANTS:-ne11 ne12 p27044 successor p29953}
# CMake hands nvcc the toolkit's include directory, but a search path from a shell profile (CPATH and friends) is
# searched first, and one naming another toolkit mixes headers: CUDA 12.1's crt/host_config.h refuses gcc 13.
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH

mkdir -p "$OUT/bin" "$OUT/logs"
F=ggml/src/ggml-cuda/mmq.cu
H=ggml/src/ggml-cuda/mmq.cuh

cd "$SRC" || exit 1
restore() { git checkout --quiet -- "$F" "$H" 2>/dev/null; pkill -P $$ 2>/dev/null; }
trap restore EXIT
git diff --quiet -- "$F" "$H" || { echo "mmq.cu/mmq.cuh are dirty: start from a clean dd266785c"; exit 1; }
grep -q 'ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne12) \* sizeof(block_q8_1_mmq);' "$F" \
    || { echo "$F does not pad the ids branch with get_J_max(ne12): this is not master with #29941"; exit 1; }
if ! grep -q 'get_J_max(113) = 64' tests/test-backend-ops.cpp; then
    git apply "$HERE/mmq-tests.patch" || { echo "mmq-tests.patch does not apply"; exit 1; }
fi

cmake -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
      -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME" -DBUILD_SHARED_LIBS=OFF \
      -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF \
      > "$OUT/logs/cmake.log" 2>&1 || { echo "cmake failed, see $OUT/logs/cmake.log"; exit 1; }

# p29953 is the only variant that edits mmq.cuh, so every MMQ instance rebuilds for it. Build it last and the
# other eight builds only recompile mmq.cu.
for v in $VARIANTS; do for e in stock debug; do
    [ -x "$OUT/bin/test-backend-ops-$v-$e" ] && continue
    git checkout --quiet -- "$F" "$H"
    python3 "$HERE/mmq-variant.py" . "$v" $([ $e = debug ] && echo --debug) > /dev/null || exit 1
    echo "building $v-$e ..."
    ninja -C "$BUILD_DIR" -j "$JOBS" test-backend-ops > "$OUT/logs/build-$v-$e.log" 2>&1 \
        || { echo "build $v-$e failed, see $OUT/logs/build-$v-$e.log"; exit 1; }
    cp "$BUILD_DIR"/bin/test-backend-ops "$OUT/bin/test-backend-ops-$v-$e"
done; done
git checkout --quiet -- "$F" "$H"

# The twelve cases of mmq-tests.patch, in its order. test_mul_mat_id(type_a, f32, n_mats, n_used, b, m, n, k)
# gives ne02 = n_mats experts, n_expert_used = n_used, ne11 = b ? 1 : n_used, ne12 = n tokens, ne01 = m rows.
CASES_ALL="p29847_b0 p29847_b1 j100_b0 j100_b1 e120_b0 e120_b1 one113 orig2040 fb65 t100 ids16 ids64 dense321"
CASES=${CASES:-$CASES_ALL}
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
    [ids16]='type_a=q4_0,type_b=f32,n_mats=256,n_used=16,b=0,m=640,n=16,k=2560,'
    [ids64]='type_a=q4_0,type_b=f32,n_mats=256,n_used=64,b=0,m=640,n=100,k=2560,'
    # MUL_MAT, not MUL_MAT_ID: the dense branch reads a padded y tile too. Short only on sm_75 among NVIDIA.
    # -p is a regex (std::regex_search), so the brackets must be escaped or [1,1] reads as a character class
    [dense321]='type_a=q2_K,type_b=f32,m=512,n=321,k=1024,bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],'
)
declare -A OP=( [dense321]=MUL_MAT )   # every other case is MUL_MAT_ID
# mode -> binary suffix, MMQ_DEBUG_ALLOC value, sanitizer yes/no
declare -A M_BIN=( [stock]=stock [memcheck]=stock [exact]=debug [guard]=debug [guard_ids_dst]=debug [guard_src1]=debug )
declare -A M_ENV=( [stock]=off [memcheck]=off [exact]=exact [guard]=guard [guard_ids_dst]=guard:ids_dst [guard_src1]=guard:src1 )
declare -A M_SAN=( [stock]=0 [memcheck]=1 [exact]=1 [guard]=0 [guard_ids_dst]=0 [guard_src1]=0 )
declare -A M_REPS=( [stock]=$REPS [memcheck]=1 [exact]=1 [guard]=$REPS [guard_ids_dst]=$REPS [guard_src1]=$REPS )

free_mib() { nvidia-smi --id="$GPU" --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
TSV=$OUT/results.tsv
[ -s "$TSV" ] || printf 'case\tvariant\tmode\trep\tpassed\tillegal_access\tmemcheck_errors\n' > "$TSV"
for c in $CASES; do for v in $VARIANTS; do for mode in $MODES; do
    bin=$OUT/bin/test-backend-ops-$v-${M_BIN[$mode]}
    [ -x "$bin" ] || { echo "missing $bin"; continue; }
    for r in $(seq 1 "${M_REPS[$mode]}"); do
        log=$OUT/logs/${c}_${v}_${mode}_r$r.log
        grep -q '^RESULT' "$log" 2>/dev/null && continue   # resumable
        while [ "$(free_mib)" -lt "$MIN_FREE_MIB" ]; do sleep 60; done
        if [ "${M_SAN[$mode]}" = 1 ]; then
            env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" MMQ_DEBUG_ALLOC="${M_ENV[$mode]}" timeout 2700 \
                "$SAN" --tool memcheck --show-backtrace no --print-limit 3 \
                "$bin" test -o "${OP[$c]:-MUL_MAT_ID}" -b CUDA0 -p "${P[$c]}" > "$log" 2>&1
        else
            env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" MMQ_DEBUG_ALLOC="${M_ENV[$mode]}" timeout 900 \
                "$bin" test -o "${OP[$c]:-MUL_MAT_ID}" -b CUDA0 -p "${P[$c]}" > "$log" 2>&1
        fi
        pass=$(grep -qE '^ *1/1 tests passed' "$log" && echo 1 || echo 0)
        ill=$(grep -c 'illegal memory access' "$log")
        errs=$(grep -oE 'ERROR SUMMARY: [0-9]+ errors?' "$log" | grep -oE '[0-9]+' | tail -1)
        printf 'RESULT\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$c" "$v" "$mode" "$r" "$pass" "$ill" "${errs:-}" >> "$log"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$c" "$v" "$mode" "$r" "$pass" "$ill" "${errs:-}" >> "$TSV"
        printf '%-10s %-10s %-14s r%s  pass=%s illegal=%s memcheck_errors=%s\n' \
            "$c" "$v" "$mode" "$r" "$pass" "$ill" "${errs:-.}"
    done
done; done; done

"$HERE/mmq-rules-table.py" "$TSV" | tee "$OUT/matrix.md"
