#!/usr/bin/env bash
# What the y-tile guard costs at compile time: mmq.cu on #29953's head (3070d927f) with the guard patch, nvcc 12.8,
# sm_120, built REPS times with the guard on and REPS times with it compiled out, interleaved so drifting host load
# hits both alike. Prints each time and the medians. One build is too noisy to quote (first run 9.8 vs 8.9 s, second
# 11.2 vs 7.9 s).
set -u
cd "$(dirname "$0")" || exit 1
HERE=$PWD
REPS=${REPS:-5}
WT=$HERE/wt-29953-guard-timing
OUT=$HERE/guard-timing-results
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
rm -rf "$WT" "$OUT"; mkdir -p "$OUT"; git -C llama.cpp worktree prune
git -C llama.cpp worktree add --quiet --detach "$WT" 3070d927f6c172242d56a2c1fb897e838e7b2632 || exit 1
trap 'cd "$HERE"; git -C llama.cpp worktree remove --force "$WT" 2>/dev/null' EXIT
cd "$WT" && git apply "$HERE/mmq-29953-y-tile-guard.patch" || exit 1
CU=ggml/src/ggml-cuda/mmq.cu
cp "$CU" "$OUT/on.cu"
sed 's/^#    define GGML_CUDA_MMQ_Y_TILE_GUARD_ON$/\/\/ off/; s/^#        define GGML_CUDA_MMQ_Y_TILE_GUARD_ON$/\/\/ off/' "$OUT/on.cu" > "$OUT/off.cu"
cmp -s "$OUT/on.cu" "$OUT/off.cu" && { echo "the off variant is identical to on: sed matched nothing"; exit 1; }
echo "host load at start (1/5/15 min): $(cut -d' ' -f1-3 /proc/loadavg), $(nproc) cores"
build() { cp "$OUT/$1.cu" "$CU"; local t0 t1; t0=$(date +%s.%N)
    "$CUDA_HOME/bin/nvcc" -Wno-deprecated-gpu-targets -std=c++17 -gencode arch=compute_120,code=sm_120 \
        -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cuda -c "$CU" -o "$OUT/mmq.o" > "$OUT/$1.log" 2>&1 || { echo "build $1 failed"; exit 1; }
    t1=$(date +%s.%N); awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}'; }
on=(); off=()
for r in $(seq 1 "$REPS"); do
    a=$(build on); b=$(build off); on+=("$a"); off+=("$b")
    printf 'rep %d: guard on %ss, off %ss\n' "$r" "$a" "$b"
done
med() { printf '%s\n' "$@" | sort -n | awk '{v[NR]=$1} END {print (NR%2 ? v[(NR+1)/2] : (v[NR/2]+v[NR/2+1])/2)}'; }
echo "median: guard on $(med "${on[@]}")s, off $(med "${off[@]}")s, over $REPS interleaved reps"
echo "host load at end (1/5/15 min): $(cut -d' ' -f1-3 /proc/loadavg)"
