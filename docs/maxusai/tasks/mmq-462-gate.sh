#!/usr/bin/env bash
# #462's retirement gate on CUDA: this branch's 903 at b11351 under the debug allocator's guard pages.
# The tree is b11351 plus this branch's ggml-cuda patches only (805, 903, 908): that is what the shipped
# libggml-cuda carries, and 001's hooks need ollama's compat.cmake, so a plain llama.cpp build cannot take the
# full series. mmq.cu/mmq.cuh are byte-identical to the full-series tree (checked by mmq-462-nvcc.sh's trees).
#   pr462      the branch as is: src1 padded by the padded y tile, ids_dst by J_best-1   -> must pass everything
#   pr462jb    positive control for guard:src1: src1 padded by J_best blocks
#   pr462ids0  positive control for guard:ids_dst: no ids_dst padding
# A pr462 pass means something only where its control faults in the same mode on the same case.
#   SM=75  GPU=1: dense321 ids16          SM=120 GPU=0: ids16 j100_b1 j100_b0   (ggml builds 120 as 120a)
# Usage: SM=75|120 [GPU=n] [PR_REF=HEAD] mmq-462-gate.sh   (from this directory; work goes to $WORK)
# Results on the CUDA host, 2026-10-13: mmq-successor-results/gate-462-sm120.tsv and gate-462-sm75.tsv.
set -u
cd "$(dirname "$0")" || exit 1
HERE=$PWD; REPO=$(git rev-parse --show-toplevel)
WORK=${WORK:-$PWD/mmq-462-work}; PR_REF=${PR_REF:-HEAD}
SM=${SM:?SM=75 or SM=120}
case $SM in
  75)  GPU=${GPU:-1}; CASES="dense321 ids16"; MIN=${MIN:-4096} ;;
  120) GPU=${GPU:-0}; CASES="ids16 j100_b1 j100_b0"; MIN=${MIN:-20480} ;;
  *)   echo "SM must be 75 or 120"; exit 2 ;;
esac
mkdir -p "$WORK"
[ -d "$WORK/b11351-pristine" ] || git clone -q --depth 1 --branch b11351 https://github.com/ggml-org/llama.cpp.git "$WORK/b11351-pristine" || exit 1
T=$WORK/tree-gate
if [ ! -d "$T" ]; then
  cp -a "$WORK/b11351-pristine" "$T" || exit 1
  for n in 805 903 908; do
    p=$(git -C "$REPO" ls-tree --name-only "$PR_REF" llama/compat/ | grep "/$n-.*\.patch$")
    git -C "$REPO" show "$PR_REF:$p" | (cd "$T" && git apply) || { echo "### $p does not apply to b11351"; exit 1; }
  done
  # one commit, so mmq-rules-gpu.sh's `git checkout -- mmq.cu mmq.cuh` restores this branch's 903, not b11351's
  git -C "$T" -c user.name=gate -c user.email=gate@localhost commit -qam "b11351 + #462 ggml-cuda patches 805 903 908"
fi
OUT=$WORK/gate-sm$SM
echo "### GATE-462 sm_$SM start $(date '+%F %T') tree $(git -C "$T" log -1 --format=%h) load $(cut -d' ' -f1-3 /proc/loadavg)"
BUILD_KINDS=debug VARIANTS="pr462 pr462jb pr462ids0" CASES="$CASES" MODES="guard_src1 guard_ids_dst" REPS=3 \
  CUDA_ARCH=$SM GPU=$GPU MIN_FREE_MIB=$MIN JOBS=${JOBS:-6} BUILD_DIR=build-462-sm$SM \
  "$HERE/mmq-rules-gpu.sh" "$T" "$OUT" || { echo "### GATE-462 sm_$SM FAILED (guarded pass)"; exit 1; }
BUILD_KINDS="stock debug" VARIANTS=pr462 CASES="$CASES" MODES="stock exact" REPS=3 \
  CUDA_ARCH=$SM GPU=$GPU MIN_FREE_MIB=$MIN JOBS=${JOBS:-6} BUILD_DIR=build-462-sm$SM \
  "$HERE/mmq-rules-gpu.sh" "$T" "$OUT" || { echo "### GATE-462 sm_$SM FAILED (stock/exact pass)"; exit 1; }
echo "### leftover GPU processes (expect none): $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -cE 'test-backend-ops|compute-sanitizer')"
echo "### GATE-462 sm_$SM DONE $(date '+%F %T') -> $OUT/results.tsv"
