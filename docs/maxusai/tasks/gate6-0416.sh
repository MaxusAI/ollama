#!/bin/bash
# Gate 6 of the v0.40.16 fold (#508) on CUDA, on the maintainer's go of 2026-11-14 ("then start the gate 6
# campaigns"). It runs after the canary. Two arms, each a cold container per model on GPU0 (device=0) with
# production's env:
#   fold        (t0416f_, l0416f_)  sync-0.40.16-fold, 0.40.13-dynres-8-ga774f7a (b12177, MLX 54a70511, no 908)
#   production  (t0416p_, l0416p_)  v0.40.13-dynres, 0.40.13-dynres-0-g85d6368, the image :11497 runs (b11963, 908)
# The production arm is a same-day, same-setup control, not :11497 itself.
# 1. Think-off cells of gemma4:31b, 26b-a4b and e4b, the GGUF vision gate. 7bd180830 stops letterboxing gemma4 images,
#    so scores and goldens may move, bounding boxes most. Compare with cmp_scored.py.
# 2. Think-on, the full ladder (16384 -> 131072), on 26b-a4b then 31b, both arms. 1b6935b22 is the first CUDA
#    flash-attention change since 908's retirement. The production arm carries 908.
# Derived from loop908-0413.sh (the same cell, restart hook and overhead).
set -u
S=/mnt/8TB_SN850X_RAID1_BTRFS/claude-scratch; R=/mnt/8TB_SN850X_RAID1_BTRFS/preflight-runs
W=$S/wt-sync0416; V=$W/docs/maxusai/vision-suite; VOL=ollama_store_8tb
IMGF=maxusai/ollama:sync-0.40.16-fold; WANTF=0.40.13-dynres-8-ga774f7a
IMGP=maxusai/ollama:v0.40.13-dynres;   WANTP=0.40.13-dynres-0-g85d6368
. $S/gpu-fit-0350.sh
OVERHEAD=8589934592; CPUS=24-31
PENV=(-e OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0 -e OLLAMA_KV_CACHE_TYPE=f16 -e OLLAMA_FORMAT_TWO_PASS=1 -e OLLAMA_NOPRUNE=true)
ts() { date '+%F %T'; }
CUR=""
cleanup() { [ -n "$CUR" ] && { docker logs "$CUR" >> "$R/$CUR-runner.log" 2>&1; docker rm -f "$CUR" >/dev/null 2>&1; CUR=""; }; }
trap 'cleanup' EXIT
trap 'echo "### GATE6-0416 INTERRUPTED $(ts)"; cleanup; trap - EXIT; exit 130' INT TERM
echo "### GATE6-0416 start $(ts) suite tree $(git -C $W rev-parse --short HEAD)"
for i in "$IMGF" "$IMGP"; do docker image inspect "$i" >/dev/null 2>&1 || { echo "### no image $i"; exit 2; }; done

# after the canary
until grep -q '### CANARY-0416 DONE\|### CANARY-0416 INTERRUPTED' $S/canary-0416.log 2>/dev/null || ! pgrep -f 'tasks/canary-0416\.sh' >/dev/null; do sleep 300; done
echo "### canary finished $(ts): $(grep -E '^### CANARY-0416 (DONE|INTERRUPTED)' $S/canary-0416.log | tail -1)"

cell() { # <image> <want> <prefix> <think on|false> <model>
  local img=$1 want=$2 prefix=$3 think=$4 m=$5 port=11587 v="" i
  wait_fit "$(need_for "$m")" "$prefix $m"
  CUR=gate6-0416; docker rm -f "$CUR" >/dev/null 2>&1
  docker run -d --name "$CUR" --gpus '"device=0"' --cpuset-cpus $CPUS -v $VOL:/root/.ollama -v "$S/jitcache:/jitcache" \
    -e CUDA_CACHE_PATH=/jitcache/nv "${PENV[@]}" -p 127.0.0.1:$port:11434 -e OLLAMA_HOST=0.0.0.0:11434 \
    -e OLLAMA_MAX_LOADED_MODELS=1 -e OLLAMA_GPU_OVERHEAD=$OVERHEAD "$img" >/dev/null || { echo "### $CUR did not start"; return 1; }
  for i in $(seq 1 90); do v=$(curl -sf --max-time 3 http://127.0.0.1:$port/api/version | sed -n 's/.*"version":"\([^"]*\)".*/\1/p'); [ -n "$v" ] && break; sleep 2; done
  [ "$v" = "$want" ] || { echo "### $CUR serves '${v:-nothing}', not $want"; cleanup; return 1; }
  printf '#!/bin/sh\ndocker restart %s >/dev/null 2>&1\nfor i in $(seq 1 90); do curl -sf --max-time 5 http://127.0.0.1:%s/api/version >/dev/null 2>&1 && exit 0; sleep 2; done\nexit 1\n' "$CUR" "$port" > "$R/restart_$CUR.sh"; chmod +x "$R/restart_$CUR.sh"
  echo "### $prefix $m think=$think start $(ts) on $img serving $v, GPU0 free $(gpu_free) MiB, load $(cut -d' ' -f1-3 /proc/loadavg)"
  if [ "$think" = on ]; then
    ( cd "$V" && RESTART_CMD="$R/restart_$CUR.sh" TAG_PREFIX="$prefix" THINK_MODES="on" MODELS="$m" ./run_engine_compare.sh http://127.0.0.1:$port >> "$R/${prefix}thinkon.log" 2>&1 )
  else
    ( cd "$V" && RESTART_CMD="$R/restart_$CUR.sh" TAG_PREFIX="$prefix" CTX_START=8192 CTX_MAX=65536 THINK_MODES="false" MODELS="$m" ./run_engine_compare.sh http://127.0.0.1:$port >> "$R/${prefix}thinkfalse.log" 2>&1 )
  fi
  local lf="$R/${prefix}think$think.log"
  echo "### $prefix $m think=$think end $(ts): suites $(grep -c 'SUITE DONE' "$lf"), NOT CONVERGED $(grep -c '^##### NOT CONVERGED' "$lf"), errors $(grep -c '^ERROR' "$lf"), launches $(docker logs "$CUR" 2>&1 | grep -oE ' -b [0-9]+ -ub [0-9]+' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
  cleanup
}
# 1. think-off, both arms per model
for m in gemma4:31b-it-q4_K_M gemma4:26b-a4b-it-q4_K_M gemma4:e4b-it-q4_K_M; do
  cell "$IMGF" "$WANTF" t0416f_ false "$m"
  cell "$IMGP" "$WANTP" t0416p_ false "$m"
done
echo "### THINK-OFF-0416 DONE $(ts)"
# 2. think-on, the full ladder, both arms per model
for m in gemma4:26b-a4b-it-q4_K_M gemma4:31b-it-q4_K_M; do
  cell "$IMGF" "$WANTF" l0416f_ on "$m"
  cell "$IMGP" "$WANTP" l0416p_ on "$m"
done
# NOT CONVERGED lines are one per ladder, not per test; count tests from the logs afterwards
echo "### NOT CONVERGED lines, fold: $(grep '^##### NOT CONVERGED' $R/l0416f_thinkon.log 2>/dev/null | grep -vc standing) / production: $(grep '^##### NOT CONVERGED' $R/l0416p_thinkon.log 2>/dev/null | grep -vc standing)"
echo "### GATE6-0416 DONE $(ts)"
