#!/bin/bash
# 908's loop-rate gate at the v0.40.5 fold (b11669), the maintainer's "go ahead" of 2026-10-16. The retirement
# register's gate: gemma4:26b-a4b full think-on suite on the full ladder (16384 -> 131072), the pin as shipped against
# the pin with 908, NOT CONVERGED counts (b11081: 1 of 27 with 908, 6 without, on sm_120); then the think-off cells of
# gemma4:31b, 26b and e4b on both arms, to set against production's (e2b moves with both halves, so it is left out).
# b11669 reworks the FA tile scheduling around 908's entries (#29435 whole-tile scheduling, #30103 race fix), so the
# b11081 counts do not carry.
#   with 908   (l0405w_, t0405w_)  the fold image sync-0.40.5-fold
#   without    (l0405n_, t0405n_)  the same image with libggml-cuda.so from the fold minus 908 (one library swapped;
#                                  every other file in the cuda_v13 payload is checked byte-identical first)
# Each arm is bench-0402.sh's cell: a cold container per model on GPU0 (device=0), production's env, 8 GiB overhead,
# run_engine_compare.sh with its restart hook. Runs after canary-0405.sh, so it never shares GPU0 with the canary.
set -u
S=/mnt/8TB_SN850X_RAID1_BTRFS/claude-scratch; R=/mnt/8TB_SN850X_RAID1_BTRFS/preflight-runs
W=$S/wt-sync0405; V=$W/docs/maxusai/vision-suite; VOL=ollama_store_8tb
IMG=maxusai/ollama:sync-0.40.5-fold; IMGN=maxusai/ollama:sync-0.40.5-fold-no908; WANT=0.40.3-dynres-12-g8761533
WN=$S/wt-sync0405-no908; OUTN=$S/no908-0405
. $S/gpu-fit-0350.sh
OVERHEAD=8589934592; CPUS=24-31
PENV=(-e OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0 -e OLLAMA_KV_CACHE_TYPE=f16 -e OLLAMA_FORMAT_TWO_PASS=1 -e OLLAMA_NOPRUNE=true)
ts() { date '+%F %T'; }
CUR=""
cleanup() { [ -n "$CUR" ] && { docker logs "$CUR" >> "$R/$CUR-runner.log" 2>&1; docker rm -f "$CUR" >/dev/null 2>&1; CUR=""; }; }
trap 'cleanup' EXIT
trap 'echo "### LOOP908-0405 INTERRUPTED $(ts)"; cleanup; trap - EXIT; exit 130' INT TERM
echo "### LOOP908-0405 start $(ts)"

# 1. the fold image
until docker image inspect "$IMG" >/dev/null 2>&1 || grep -q 'BUILD-0405 FAILED' $S/build-0405.log; do sleep 300; done
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "### fold build failed"; exit 2; }

# 2. the no-908 payload: the fold's tree minus 908, only the cuda_v13 llama-server stage
if ! docker image inspect "$IMGN" >/dev/null 2>&1; then
  [ -d "$WN" ] || git -C /opt/github/MaxusAI/ollama worktree add -q --detach "$WN" "$(git -C $W rev-parse HEAD)" || exit 2
  rm -f "$WN/llama/compat/908-revert-fattn-mma-gemma4-tiling.patch"
  ( cd "$WN" && . scripts/env.sh && rm -rf "$OUTN" && mkdir -p "$OUTN" && echo "### no-908 build start $(ts) head $(git rev-parse --short HEAD), patches: $(ls llama/compat/*.patch | xargs -n1 basename | cut -c1-3 | tr '\n' ' ')" \
    && docker buildx build --builder bigdisk --platform=linux/amd64 ${OLLAMA_COMMON_BUILD_ARGS} -f Dockerfile \
         --target publish-llama-server-cuda_v13 --output type=local,dest="$OUTN" . > $S/build-0405-no908.log 2>&1 ) \
    || { echo "### no-908 build failed: $(tail -3 $S/build-0405-no908.log)"; exit 2; }
  echo "### no-908 build done $(ts)"
  LIBDIR=$(docker run --rm --entrypoint sh "$IMG" -c 'dirname $(ls /usr/lib/ollama/cuda_v13/libggml-cuda.so /lib/ollama/cuda_v13/libggml-cuda.so 2>/dev/null | head -1)')
  [ -n "$LIBDIR" ] || { echo "### cannot find cuda_v13/libggml-cuda.so in $IMG"; exit 2; }
  rm -rf "$S/fold-0405-cuda_v13"; mkdir -p "$S/fold-0405-cuda_v13"
  cid=$(docker create "$IMG"); docker cp "$cid:$LIBDIR/." "$S/fold-0405-cuda_v13/" >/dev/null; docker rm "$cid" >/dev/null
  diffs=$(cd "$OUTN/lib/ollama/cuda_v13" && for f in $(find . -type f | sort); do cmp -s "$f" "$S/fold-0405-cuda_v13/$f" || echo "$f"; done)
  echo "### cuda_v13 files that differ from the fold image: $(echo $diffs)"
  [ "$(echo $diffs)" = "./libggml-cuda.so" ] || { echo "### expected exactly libggml-cuda.so to differ; not swapping"; exit 2; }
  printf 'FROM %s\nCOPY libggml-cuda.so %s/libggml-cuda.so\n' "$IMG" "$LIBDIR" > "$OUTN/Dockerfile.no908"
  docker build -q -f "$OUTN/Dockerfile.no908" -t "$IMGN" "$OUTN/lib/ollama/cuda_v13" >/dev/null || { echo "### swap image build failed"; exit 2; }
  echo "### $IMGN built $(ts): libggml-cuda.so $(sha256sum $OUTN/lib/ollama/cuda_v13/libggml-cuda.so | cut -c1-16) (fold's $(sha256sum $S/fold-0405-cuda_v13/libggml-cuda.so | cut -c1-16))"
fi

# 3. after the canary
until grep -q '### CANARY-0405 DONE\|### CANARY-0405 INTERRUPTED' $S/canary-0405.log 2>/dev/null || ! pgrep -of "bash $S/canary-0405.sh" >/dev/null; do sleep 300; done
echo "### canary finished $(ts): $(grep -E '^### CANARY-0405 (DONE|INTERRUPTED)' $S/canary-0405.log | tail -1)"

cell() { # <image> <prefix> <think on|false> <model>
  local img=$1 prefix=$2 think=$3 m=$4 port=11586 v="" i
  wait_fit "$(need_for "$m")" "$prefix $m"
  CUR=loop908-0405; docker rm -f "$CUR" >/dev/null 2>&1
  docker run -d --name "$CUR" --gpus '"device=0"' --cpuset-cpus $CPUS -v $VOL:/root/.ollama -v "$S/jitcache:/jitcache" \
    -e CUDA_CACHE_PATH=/jitcache/nv "${PENV[@]}" -p 127.0.0.1:$port:11434 -e OLLAMA_HOST=0.0.0.0:11434 \
    -e OLLAMA_MAX_LOADED_MODELS=1 -e OLLAMA_GPU_OVERHEAD=$OVERHEAD "$img" >/dev/null || { echo "### $CUR did not start"; return 1; }
  for i in $(seq 1 90); do v=$(curl -sf --max-time 3 http://127.0.0.1:$port/api/version | sed -n 's/.*"version":"\([^"]*\)".*/\1/p'); [ -n "$v" ] && break; sleep 2; done
  [ "$v" = "$WANT" ] || { echo "### $CUR serves '${v:-nothing}', not $WANT"; cleanup; return 1; }
  printf '#!/bin/sh\ndocker restart %s >/dev/null 2>&1\nfor i in $(seq 1 90); do curl -sf --max-time 5 http://127.0.0.1:%s/api/version >/dev/null 2>&1 && exit 0; sleep 2; done\nexit 1\n' "$CUR" "$port" > "$R/restart_$CUR.sh"; chmod +x "$R/restart_$CUR.sh"
  echo "### $prefix $m think=$think start $(ts) on $img, GPU0 free $(gpu_free) MiB"
  if [ "$think" = on ]; then
    ( cd "$V" && RESTART_CMD="$R/restart_$CUR.sh" TAG_PREFIX="$prefix" THINK_MODES="on" MODELS="$m" ./run_engine_compare.sh http://127.0.0.1:$port >> "$R/${prefix}thinkon.log" 2>&1 )
  else
    ( cd "$V" && RESTART_CMD="$R/restart_$CUR.sh" TAG_PREFIX="$prefix" CTX_START=8192 CTX_MAX=65536 THINK_MODES="false" MODELS="$m" ./run_engine_compare.sh http://127.0.0.1:$port >> "$R/${prefix}thinkfalse.log" 2>&1 )
  fi
  local lf="$R/${prefix}think$think.log"
  echo "### $prefix $m think=$think end $(ts): suites $(grep -c 'SUITE DONE' "$lf"), NOT CONVERGED $(grep -c '^##### NOT CONVERGED' "$lf"), errors $(grep -c '^ERROR' "$lf"), launches $(docker logs "$CUR" 2>&1 | grep -oE ' -b [0-9]+ -ub [0-9]+' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
  cleanup
}
# 4. the loop-rate arms, then the think-off cells
cell "$IMG"  l0405w_ on gemma4:26b-a4b-it-q4_K_M
cell "$IMGN" l0405n_ on gemma4:26b-a4b-it-q4_K_M
for m in gemma4:31b-it-q4_K_M gemma4:26b-a4b-it-q4_K_M gemma4:e4b-it-q4_K_M; do
  cell "$IMG"  t0405w_ false "$m"
  cell "$IMGN" t0405n_ false "$m"
done
echo "### NOT CONVERGED with 908: $(grep '^##### NOT CONVERGED' $R/l0405w_thinkon.log | grep -vc standing) / without: $(grep '^##### NOT CONVERGED' $R/l0405n_thinkon.log | grep -vc standing)"
echo "### LOOP908-0405 DONE $(ts)"
