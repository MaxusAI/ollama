#!/usr/bin/env bash
# A compile-time guard for MMQ's y tile on top of llama.cpp#29953's head (3070d927f): material for upstream, GPU-free.
#
# mmq-29953-y-tile-guard.patch routes the src1 padding and the shared-memory y tile through one helper,
# ggml_cuda_mmq_get_nbytes_y_tile(), and asserts at compile time -- for every config table and type -- that the
# helper covers the y load loop's extent, written from the loop with exact ceiling division. Five nvcc builds of
# mmq.cu for sm_120, each scored by WHY it failed, not just whether:
#   1. the head as is, unpatched          -> must compile (baseline time)
#   2. the head with the patch            -> must compile
#   3. helper shrunk to J blocks          -> must FAIL by the guard, naming every table (the published #29953's rule)
#   4. helper rounded to one warp's ints  -> must FAIL by the guard, naming every table
#   5. one blackwell config at 192 threads-> must FAIL by the guard naming ONLY blackwell: GGML_PAD is a bit mask
#                                            that assumes a power-of-two granularity, the tables only assert
#                                            nthreads % 32 == 0, and the guard's exact arithmetic sees the gap
# nvcc's front end stops at 100 errors by default, which cut mutation 3's table list to the first five tables;
# --error_limit lifts that so every table that fails is named.
# Usage: mmq-29953-guard-verify.sh   (from its own directory, with a llama.cpp clone at ./llama.cpp that has
#        3070d927f fetched; results go to ./verify-29953-guard-results)
set -u
cd "$(dirname "$0")" || exit 1
HERE=$PWD
PATCH=$HERE/mmq-29953-y-tile-guard.patch
WT=$HERE/wt-29953-guard-verify
OUT=$HERE/verify-29953-guard-results
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
HEAD29953=3070d927f6c172242d56a2c1fb897e838e7b2632
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
rm -rf "$OUT"; mkdir -p "$OUT"
echo "host load at start (1/5/15 min): $(cut -d' ' -f1-3 /proc/loadavg), $(nproc) cores"

rm -rf "$WT"; git -C llama.cpp worktree prune
git -C llama.cpp worktree add --quiet --detach "$WT" "$HEAD29953" || exit 1
trap 'cd "$HERE"; git -C llama.cpp worktree remove --force "$WT" 2>/dev/null' EXIT
cd "$WT" || exit 1
CU=ggml/src/ggml-cuda/mmq.cu; CUH=ggml/src/ggml-cuda/mmq.cuh; BW=ggml/src/ggml-cuda/mmq-config-blackwell.cuh

build() { # <label> <expect ok|fail> [expected table list, space-separated, sorted]
    local label=$1 expect=$2 want_tables=${3:-}
    local log="$OUT/$label.log" t0 t1 rc
    t0=$(date +%s.%N)
    "$CUDA_HOME/bin/nvcc" -Wno-deprecated-gpu-targets -std=c++17 -gencode arch=compute_120,code=sm_120 \
        -Xcudafe --error_limit=100000 \
        -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cuda -c "$CU" -o "$OUT/mmq.o" > "$log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    local got=ok; [ $rc -ne 0 ] && got=fail
    local tables; tables=$(grep -oE 'in the [a-z0-9_]+ table' "$log" | sed 's/in the //;s/ table//' | sort -u | tr '\n' ' ' | sed 's/ $//')
    # count errors, not lines: nvcc prints each failed static_assert on two
    local nerr; nerr=$(grep -cE ': error' "$log")
    local nguard; nguard=$(grep -c ': error: static assertion failed with "MMQ: the y tile helper' "$log")
    local verdict=UNEXPECTED
    if [ "$got" = "$expect" ]; then
        if [ "$expect" = ok ]; then verdict=as-expected
        elif [ "$nguard" -gt 0 ] && [ "$nerr" -eq "$nguard" ] && [ "$tables" = "$want_tables" ]; then verdict=as-expected
        else verdict=WRONG-REASON
        fi
    fi
    printf '%-34s rc=%-3s %-4s want=%-4s %-13s %5.1fs\n' "$label" "$rc" "$got" "$expect" "$verdict" "$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b-a}')"
    [ "$got" = fail ] && printf '    errors=%s, from the guard=%s, tables named: %s\n' "$nerr" "$nguard" "${tables:-none}"
    [ "$verdict" = as-expected ]
}

ALL="ampere blackwell cdna gcn pascal_dp4a pascal_older rdna2 rdna3 rdna3_5 rdna4"
ok=1
build "1-head-unpatched" ok || ok=0

git apply "$PATCH" || { echo "FAIL: the patch does not apply to $HEAD29953"; exit 1; }
echo "patch applies to a pristine $HEAD29953"
cp "$CU" "$OUT/mmq.cu.patched"; cp "$CUH" "$OUT/mmq.cuh.patched"; cp "$BW" "$OUT/blackwell.orig"
build "2-head-with-guard" ok || ok=0

swap() { # <file> <old> <new>: exact one-occurrence string swap, so a mutation cannot fail for a typo of mine
    python3 - "$1" "$2" "$3" <<'PY' || exit 1
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if s.count(old) != 1:
    sys.exit("mutation anchor found %d times in %s, expected once: %r" % (s.count(old), p, old))
open(p, "w").write(s.replace(old, new))
PY
}
HELPER='    return GGML_PAD(config.J*sizeof(block_q8_1_mmq), config.nthreads*sizeof(int));'

swap "$CUH" "$HELPER" '    return config.J*sizeof(block_q8_1_mmq);'
build "3-helper-shrunk-to-J-blocks" fail "$ALL" || ok=0
cp "$OUT/mmq.cuh.patched" "$CUH"

swap "$CUH" "$HELPER" '    return GGML_PAD(config.J*sizeof(block_q8_1_mmq), 32*sizeof(int));'
build "4-helper-rounded-to-one-warp" fail "$ALL" || ok=0
cp "$OUT/mmq.cuh.patched" "$CUH"

# the whole first CASE line with 256 threads: the type alone repeats once per J, so it is not a unique anchor
first=$(grep -m1 -E '^ *CASE\(GGML_TYPE_[A-Z0-9_]+, *256,' "$BW")
swap "$BW" "$first" "${first/256,/192,}"
echo "mutation 5 rewrote: $first -> ${first/256,/192,}"
build "5-blackwell-config-at-192-threads" fail "blackwell" || ok=0
cp "$OUT/blackwell.orig" "$BW"

sed -i 's/^#    define GGML_CUDA_MMQ_Y_TILE_GUARD_ON$/\/\/ guard disabled for timing/; s/^#        define GGML_CUDA_MMQ_Y_TILE_GUARD_ON$/\/\/ guard disabled for timing/' "$CU"
build "6-guard-compiled-out" ok || ok=0

echo "VERIFY29953GUARD COMPLETE $(date -u +%FT%TZ) all_as_expected=$ok"
