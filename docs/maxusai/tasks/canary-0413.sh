#!/bin/bash
# Gate 5 of the v0.40.13 fold (#501) on CUDA (sm_120a), under the maintainer's "go ahead on your own" of 2026-11-03:
# new profiles cuda-0-40-13 and mlx-cuda-0-40-13 (rows carried from 0-40-5 as candidates, then measured), on a
# production-like canary of the fold image sync-0.40.13-fold (build-0413.sh). Derived from canary-0405.sh, without
# its Nemotron 3.5 pulls (the models are in the store).
# Production-like: --gpus all as :11497 (GPU1 is a sm_75 2080 Ti), production's env, one model at a time.
# Never :11497. GPU0 is shared with production, so each heavy step waits for production to be idle (15 min, nothing
# loaded; at most 3 h, then GPU0's own fit check alone) and for the model's footprint to fit.
# Harness: this fold's worktree. The interim stamp 0.40.3-dynres-N is claimed by the 0-40-2 profiles, so measure and
# preflight read a run-local expectations copy without them, plus 0-40-13 profiles cloned from 0-40-5 with the new
# pins and a pattern admitting the interim stamp (exp-0413.py).
# MLX PTX cache: one dir per GPU architecture and MLX pin (a new one, so the first MLX request pays the JIT).
set -u
S=/mnt/8TB_SN850X_RAID1_BTRFS/claude-scratch; R=/mnt/8TB_SN850X_RAID1_BTRFS/preflight-runs
W=$S/wt-sync0413; P=$W/docs/maxusai/vision-suite/preflight
IMG=${IMG:-maxusai/ollama:sync-0.40.13-fold}; WANT=${WANT:-0.40.3-dynres-20-g81dda7d}
RUN=${RUN:-canary-g81dda7d}
VOL=ollama_store_8tb; STORE=/mnt/8TB_SN850X_RAID1_BTRFS/opt/ollama/.ollama
CUR=ollama-canary-0413; PORT=11533
PTX=/jitcache/mlx-ptx-sm120-065ec468
EXP=$R/expectations-0413-$RUN.toml
. $S/gpu-fit-0350.sh
ts() { date '+%F %T'; }
PROD=$(docker ps --format '{{.Names}}' | grep -E '^ollama-0\.40\.' | head -1)
prod_idle() { [ "$(docker logs --since 15m "$PROD" 2>&1 | grep -cE '"/api/(generate|chat)"|"/v1/chat')" = 0 ] \
  && [ "$(curl -s --max-time 5 http://127.0.0.1:11497/api/ps | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("models",[])))' 2>/dev/null)" = 0 ]; }
wait_prod() { local n=0; until prod_idle; do [ $n -ge 36 ] && { echo "### production not idle for 3 h; going on GPU0's fit check alone $(ts)"; return; }
  [ $((n % 6)) -eq 0 ] && echo "### waiting for production ($PROD) to be idle 15 min with nothing loaded: $1 $(ts)"; n=$((n+1)); sleep 300; done
  echo "### production idle: $1 $(ts)"; }
unload() { for m in $(curl -s --max-time 5 http://127.0.0.1:$PORT/api/ps | python3 -c 'import json,sys; print(" ".join(m["name"] for m in json.load(sys.stdin).get("models",[])))'); do
  curl -s --max-time 120 http://127.0.0.1:$PORT/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null; done; }
cleanup() { docker logs "$CUR" > "$R/canary-0413-$RUN-runner.log" 2>&1; docker rm -f "$CUR" >/dev/null 2>&1
  echo "### teardown $(ts) GPU0 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0); ollama procs on GPU0: $(nvidia-smi -i 0 --query-compute-apps=pid,process_name --format=csv,noheader | grep -c 'ollama\|llama-server')"; }

echo "### CANARY-0413 $RUN start $(ts)"
until docker image inspect "$IMG" >/dev/null 2>&1 || grep -q 'BUILD-0413 FAILED' $S/build-0413.log; do sleep 300; done
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "### build failed: $(grep '^###' $S/build-0413.log | tail -2 | tr '\n' ' ')"; exit 2; }
echo "### image ready $(ts): $(docker image inspect $IMG --format '{{.Id}} {{.Created}}')"
python3 $S/exp-0413.py "$P/expectations.toml" "$EXP" "$WANT" || { echo "### could not write $EXP"; exit 2; }

wait_prod "canary start"
wait_fit "$(need_for nemotron3:33b-q8)" "gate 5 $RUN"
trap 'cleanup; exit' EXIT
trap 'echo "### CANARY-0413 INTERRUPTED"; cleanup; trap - EXIT; exit 130' INT TERM
docker rm -f "$CUR" >/dev/null 2>&1
docker run -d --name "$CUR" --gpus all -v $VOL:/root/.ollama -v "$S/jitcache:/jitcache" \
  -e CUDA_CACHE_PATH=/jitcache/nv -e MLX_PTX_CACHE_DIR=$PTX \
  -e OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0 -e OLLAMA_KV_CACHE_TYPE=f16 -e OLLAMA_FORMAT_TWO_PASS=1 \
  -p 127.0.0.1:$PORT:11434 -e OLLAMA_HOST=0.0.0.0:11434 -e OLLAMA_NOPRUNE=true \
  -e OLLAMA_MAX_LOADED_MODELS=1 "$IMG" >/dev/null
v=""; for i in $(seq 1 90); do v=$(curl -sf --max-time 3 http://127.0.0.1:$PORT/api/version | sed -n 's/.*"version":"\([^"]*\)".*/\1/p'); [ -n "$v" ] && break; sleep 2; done
[ "$v" = "$WANT" ] || { echo "### canary serves '${v:-nothing}', not $WANT"; exit 2; }
echo "### CANARY up $(ts) serving $v; devices: $(docker logs $CUR 2>&1 | grep -c 'inference compute'), fold head $(git -C $W rev-parse --short HEAD)"
echo "### payload: cuda_v13 $(docker exec $CUR sh -c 'ls /usr/lib/ollama/cuda_v13 | wc -l') files, mlx_cuda_v13 $(docker exec $CUR sh -c 'ls /usr/lib/ollama/mlx_cuda_v13 2>/dev/null | wc -l') files"
python3 "$W/docs/maxusai/tools/host_profile.py" --out "$R/hp-cuda-0413-$RUN" --label "$(head -1 ~/.config/collab/identity)" \
  --fs models="$STORE/models" --ollama http://127.0.0.1:$PORT > "$R/hp-cuda-0413-$RUN.log" 2>&1 \
  && echo "### host profile: $R/hp-cuda-0413-$RUN/host-profile.json" || echo "### host profile capture FAILED"
cd "$P" || exit 2

# 1. the ladders, one arch at a time, each printed as the paste-ready block
python3 -c 'import sys,tomllib; e=tomllib.load(open(sys.argv[1],"rb"))
for p in ("cuda-0-40-13","mlx-cuda-0-40-13"):
    for a,v in e["expect"][p].items(): print(p, a, v["model"])' "$EXP" | while read -r prof arch model; do
  wait_prod "ladder $prof $arch"; wait_fit "$(need_for "$model")" "ladder $model"
  echo "### LADDER $prof $arch $model $(ts)"
  python3 measure_ladder.py --host http://127.0.0.1:$PORT --model "$model" --profile "$prof" --arch "$arch" \
    --container "$CUR" --expectations "$EXP" --out "$R/ladder-0413-$RUN-$prof-$arch.toml" > "$R/ladder-0413-$RUN-$prof-$arch.log" 2>&1
  echo "### LADDER $prof $arch rc=$? -> $R/ladder-0413-$RUN-$prof-$arch.toml"
  unload
done

# 2. preflight on each platform, against the carried rows
rc_all=0
for plat in cuda mlx-cuda; do
  wait_prod "preflight $plat"; wait_fit "$(need_for nemotron3:33b-q8)" "preflight $plat"
  OUTJ=$R/preflight-$plat-0413-$RUN.json
  ./preflight.py --host http://127.0.0.1:$PORT --platform $plat --image-tag "$IMG" --container "$CUR" \
    --expectations "$EXP" --out "$OUTJ" > "$R/preflight-$plat-0413-$RUN.log" 2>&1
  rc=$?; [ $rc -ne 0 ] && rc_all=$rc
  echo "### PREFLIGHT-0413 $plat $RUN rc=$rc $(ts): $(grep -E 'ERROR=|PASS=' $R/preflight-$plat-0413-$RUN.log | tr -s ' ') $(grep -oE '(ERR|FAIL) +[a-z_]+ \[[a-z0-9_]+\][^:]*: [^ ]+' $R/preflight-$plat-0413-$RUN.log | tr '\n' ';')"
  unload
done
echo "### cudaMemcpy errors: $(docker logs $CUR 2>&1 | grep -c 'cudaMemcpy(data, buf.data'), OOMs: $(docker logs $CUR 2>&1 | grep -c 'out of memory')"
echo "### CANARY-0413 DONE rc=$rc_all $(ts)"
exit $rc_all
