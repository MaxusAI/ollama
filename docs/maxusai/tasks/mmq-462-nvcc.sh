#!/usr/bin/env bash
# #462's GPU-free check: does 903-as-#29953-plus-guard build at b11351, and does its guard fire? (nvcc, no GPU)
#   A. main's series (today's 903)      sm_120a, sm_75 -> must compile (control)
#   B. this branch's series             sm_120a, sm_75 -> must compile, the guard's static_asserts pass
#   C. B with the y-tile helper shrunk to J blocks (#29953's first published rule)
#                                       sm_120a        -> must FAIL, every error from the guard
# Then the patched mmq.cu/mmq.cuh are diffed against llama.cpp fc9ce6b9d (#29953 as merged).
# The compat series is applied the way cmake/apply-git-patches.cmake does: every llama/compat/*.patch, sorted.
# Usage: MAIN_REF=origin/main PR_REF=HEAD mmq-462-nvcc.sh   (from this directory; work goes to $WORK)
# Results on the CUDA host, 2026-10-13: mmq-successor-results/verify-462-nvcc.txt.
set -u
cd "$(dirname "$0")" || exit 1
REPO=$(git rev-parse --show-toplevel)
WORK=${WORK:-$PWD/mmq-462-work}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
MAIN_REF=${MAIN_REF:-origin/main}; PR_REF=${PR_REF:-HEAD}
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
mkdir -p "$WORK"; OUT=$WORK/nvcc-results; rm -rf "$OUT"; mkdir -p "$OUT"
[ -d "$WORK/b11351-pristine" ] || git clone -q --depth 1 --branch b11351 https://github.com/ggml-org/llama.cpp.git "$WORK/b11351-pristine" || exit 1
mkdir -p "$WORK/ref"
for f in mmq.cu mmq.cuh; do
  [ -s "$WORK/ref/fc9ce6b9d-$f" ] || gh api -H 'Accept: application/vnd.github.raw' \
    "repos/ggml-org/llama.cpp/contents/ggml/src/ggml-cuda/$f?ref=fc9ce6b9d" > "$WORK/ref/fc9ce6b9d-$f" || exit 1
done
echo "load at start: $(cut -d' ' -f1-3 /proc/loadavg); $("$CUDA_HOME/bin/nvcc" --version | tail -1)"
echo "main=$(git -C "$REPO" rev-parse --short=9 "$MAIN_REF") pr=$(git -C "$REPO" rev-parse --short=9 "$PR_REF") b11351=$(git -C "$WORK/b11351-pristine" rev-parse --short=9 HEAD)"

series() { # <ref> <dir>
  rm -rf "$2"; mkdir -p "$2"
  for p in $(git -C "$REPO" ls-tree --name-only "$1" llama/compat/ | grep '\.patch$' | sort); do
    git -C "$REPO" show "$1:$p" > "$2/$(basename "$p")"; done
}
tree() { # <label> <seriesdir>
  local t=$WORK/tree-$1
  rm -rf "$t"; cp -a "$WORK/b11351-pristine" "$t" || return 1
  for p in $(ls "$2"/*.patch | sort); do
    (cd "$t" && git apply "$p") || { echo "  $1: $(basename "$p") DOES NOT APPLY"; return 1; }
  done
  echo "  $1: $(ls "$2"/*.patch | wc -l) patches applied in order: $(ls "$2" | sed 's/-.*//' | tr '\n' ' ')"
}
build() { # <label> <tree> <arch> <expect ok|fail>
  local log=$OUT/$1-sm$3.log t0 rc
  t0=$(date +%s)
  (cd "$2" && nice -n 19 "$CUDA_HOME/bin/nvcc" -Wno-deprecated-gpu-targets -std=c++17 \
     -gencode arch=compute_$3,code=sm_$3 -Xcudafe --error_limit=100000 \
     -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cuda -c ggml/src/ggml-cuda/mmq.cu -o "$OUT/$1.o") > "$log" 2>&1
  rc=$?
  local got=ok; [ $rc -ne 0 ] && got=fail
  local nerr nguard tables
  nerr=$(grep -cE ': error' "$log")
  nguard=$(grep -c 'static assertion failed with "MMQ: the y tile helper' "$log")
  tables=$(grep -oE 'in the [a-z0-9_]+ table' "$log" | sed 's/in the //;s/ table//' | sort -u | tr '\n' ' ')
  local v=UNEXPECTED
  if [ $got = "$4" ]; then
    if [ "$4" = ok ]; then v=as-expected
    elif [ "$nguard" -gt 0 ] && [ "$nerr" -eq "$nguard" ]; then v=as-expected
    else v=WRONG-REASON; fi
  fi
  printf '%-18s sm_%-5s rc=%-3s %-4s want=%-4s %-13s %4ss' "$1" "$3" "$rc" "$got" "$4" "$v" "$(( $(date +%s)-t0 ))"
  [ $got = fail ] && printf '  errors=%s guard=%s tables: %s' "$nerr" "$nguard" "${tables:-none}"
  echo
  [ $v = as-expected ]
}
ok=1
series "$MAIN_REF" "$WORK/series-main"; series "$PR_REF" "$WORK/series-pr"
tree main "$WORK/series-main" || ok=0
tree pr "$WORK/series-pr" || ok=0
grep -q 'ggml_cuda_mmq_get_nbytes_y_tile' "$WORK/tree-pr/ggml/src/ggml-cuda/mmq.cuh" \
  && echo "  pr tree carries the guard helper" || { echo "  pr tree has NO guard helper"; ok=0; }
build A-main "$WORK/tree-main" 120a ok || ok=0
build A-main "$WORK/tree-main" 75   ok || ok=0
build B-pr   "$WORK/tree-pr"   120a ok || ok=0
build B-pr   "$WORK/tree-pr"   75   ok || ok=0
rm -rf "$WORK/tree-mut"; cp -a "$WORK/tree-pr" "$WORK/tree-mut"
python3 - "$WORK/tree-mut/ggml/src/ggml-cuda/mmq.cuh" <<'PY' || ok=0
import sys
p = sys.argv[1]; s = open(p).read()
old = "return GGML_PAD(config.J*sizeof(block_q8_1_mmq), config.nthreads*sizeof(int));"
if s.count(old) != 1:
    sys.exit("mutation anchor found %d times" % s.count(old))
open(p, "w").write(s.replace(old, "return config.J*sizeof(block_q8_1_mmq);"))
print("  mutation C: helper -> J blocks")
PY
build C-J-blocks "$WORK/tree-mut" 120a fail || ok=0
for f in mmq.cu mmq.cuh; do
  diff -u "$WORK/ref/fc9ce6b9d-$f" "$WORK/tree-pr/ggml/src/ggml-cuda/$f" > "$OUT/diff-fc9ce6b9d-vs-pr-$f.txt"
  echo "  $f vs fc9ce6b9d (#29953 as merged): +$(grep -c '^+[^+]' "$OUT/diff-fc9ce6b9d-vs-pr-$f.txt") -$(grep -c '^-[^-]' "$OUT/diff-fc9ce6b9d-vs-pr-$f.txt") lines"
done
rm -rf "$WORK/tree-mut" "$WORK/tree-main" "$OUT"/*.o
[ $ok = 1 ] && echo "### MMQ-462-NVCC PASS $(date '+%F %T')" || echo "### MMQ-462-NVCC FAIL $(date '+%F %T')"
