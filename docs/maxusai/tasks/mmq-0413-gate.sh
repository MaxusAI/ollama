#!/usr/bin/env bash
# 903's guard-page gate at the v0.40.13 fold (b11963), derived from mmq-0405-gate.sh, whose notes follow. Changes:
#   - b11963 instead of b11669, taken as a detached worktree of a local llama.cpp clone ($LLAMA_SRC) when there is
#     one, else cloned. At b11963 upstream already pads src1 by GGML_PAD(J*144, nthreads*4) (c49edf096); 903 is
#     re-cut to report that through ggml_cuda_mmq_get_nbytes_y_tile() and to guard it with static_asserts.
#   - mmq-rules-gpu-0413.sh / mmq-variant-0413.py: the same three variants, with the debug rewrite re-anchored on
#     c49edf096's prec_src1 line, every mutation checked in the source and in the binary, and three fused
#     gate/up+GLU cases (MUL_MAT_VEC_FUSION) from b11963's own tests.
#
# The tree is the pin plus this branch's ggml-cuda patches only (805, 903, 908): that is what the shipped
# libggml-cuda carries, and 001's hooks need ollama's compat.cmake, so a plain llama.cpp build cannot take the
# full series.
#   pr462      the branch as is: src1 padded by the padded y tile, ids_dst by J_best-1   -> must pass everything
#   pr462jb    positive control for guard:src1: src1 padded by J_best blocks
#   pr462ids0  positive control for guard:ids_dst: no ids_dst padding
# A pr462 pass means something only where its control faults in the same mode on the same case.
#   SM=75  GPU=1: dense321 ids16 + fused     SM=120 GPU=0: ids16 j100_b1 j100_b0 + fused  (ggml builds 120 as 120a)
# Usage: SM=75|120 [GPU=n] [PR_REF=HEAD] [WORK=dir] [LLAMA_SRC=clone] mmq-0413-gate.sh
set -u
cd "$(dirname "$0")" || exit 1
HERE=$PWD; REPO=$(git rev-parse --show-toplevel)
WORK=${WORK:-$PWD/mmq-0413-work}; PR_REF=${PR_REF:-HEAD}; PIN=b11963
LLAMA_SRC=${LLAMA_SRC:-/mnt/8TB_SN850X_RAID1_BTRFS/claude-scratch/llama.cpp-src}
SM=${SM:?SM=75 or SM=120}
FUSED="fu_dense17 fu_id17b fu_id2048"
case $SM in
  75)  GPU=${GPU:-1}; CASES=${CASES:-"dense321 ids16 $FUSED"}; MIN=${MIN:-4096} ;;
  120) GPU=${GPU:-0}; CASES=${CASES:-"ids16 j100_b1 j100_b0 $FUSED"}; MIN=${MIN:-20480} ;;
  *)   echo "SM must be 75 or 120"; exit 2 ;;
esac
mkdir -p "$WORK"
T=$WORK/tree-gate
if [ ! -d "$T" ]; then
  if git -C "$LLAMA_SRC" rev-parse -q --verify "$PIN^{commit}" >/dev/null 2>&1; then
    git -C "$LLAMA_SRC" worktree add -q --detach "$T" "$PIN" || exit 1
  else
    git clone -q --depth 1 --branch "$PIN" https://github.com/ggml-org/llama.cpp.git "$T" || exit 1
  fi
  for n in 805 903 908; do
    p=$(git -C "$REPO" ls-tree --name-only "$PR_REF" llama/compat/ | grep "/$n-.*\.patch$")
    git -C "$REPO" show "$PR_REF:$p" | (cd "$T" && git apply) || { echo "### $p does not apply to $PIN"; exit 1; }
  done
  # one commit, so mmq-rules-gpu-0413.sh's `git checkout -- mmq.cu mmq.cuh` restores this branch's 903, not the pin's
  git -C "$T" -c user.name=gate -c user.email=gate@localhost commit -qam \
    "$PIN + the v0.40.13 fold's ggml-cuda patches 805 903 908 ($(git -C "$REPO" rev-parse --short "$PR_REF"))"
fi
OUT=$WORK/gate-sm$SM
echo "### GATE-0413 sm_$SM start $(date '+%F %T') tree $(git -C "$T" log -1 --format=%h) on $(git -C "$T" log -1 --format=%h HEAD^) fold $(git -C "$REPO" rev-parse --short "$PR_REF") load $(cut -d' ' -f1-3 /proc/loadavg)"
BUILD_KINDS=debug VARIANTS="pr462 pr462jb pr462ids0" CASES="$CASES" MODES="guard_src1 guard_ids_dst" REPS=3 \
  CUDA_ARCH=$SM GPU=$GPU MIN_FREE_MIB=$MIN JOBS=${JOBS:-6} BUILD_DIR=build-0413-sm$SM \
  "$HERE/mmq-rules-gpu-0413.sh" "$T" "$OUT" || { echo "### GATE-0413 sm_$SM FAILED (guarded pass)"; exit 1; }
BUILD_KINDS="stock debug" VARIANTS=pr462 CASES="$CASES" MODES="stock exact" REPS=3 \
  CUDA_ARCH=$SM GPU=$GPU MIN_FREE_MIB=$MIN JOBS=${JOBS:-6} BUILD_DIR=build-0413-sm$SM \
  "$HERE/mmq-rules-gpu-0413.sh" "$T" "$OUT" || { echo "### GATE-0413 sm_$SM FAILED (stock/exact pass)"; exit 1; }
echo "### fused kernel reached (MMQ_FUSED 1 lines per case, guard_src1 r1 of pr462):"
for c in $CASES; do
  f=$OUT/logs/${c}_pr462_guard_src1_r1.log
  [ -f "$f" ] && echo "    $c: $(grep -o 'MMQ_FUSED 1' "$f" | wc -l) fused / $(grep -o 'MMQ_FUSED [01]' "$f" | wc -l) MMQ calls"
done
echo "### leftover GPU processes (expect none): $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -cE 'test-backend-ops|compute-sanitizer')"
echo "### GATE-0413 sm_$SM DONE $(date '+%F %T') -> $OUT/results.tsv"
