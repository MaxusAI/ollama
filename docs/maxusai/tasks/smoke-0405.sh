#!/bin/bash
# Nemotron 3.5 on the v0.40.5 fold image: load + one text and one image request per model, after the stock-client pulls
# (pull-0405-stock.sh). Production-like container (--gpus all, production's env, one model at a time), never :11497;
# each model waits for production to be idle and for its GPU0 room. Then relaunches loop908-0405.sh, which this run
# holds back so the two never share GPU0.
set -u
S=/mnt/8TB_SN850X_RAID1_BTRFS/claude-scratch; R=/mnt/8TB_SN850X_RAID1_BTRFS/preflight-runs
IMG=maxusai/ollama:sync-0.40.5-fold; WANT=0.40.3-dynres-12-g8761533; CUR=ollama-smoke-0405; PORT=11534
P=$S/wt-sync0405/docs/maxusai/vision-suite/preflight; IMGF=$P/ladderimgs/1024x576.png
. $S/gpu-fit-0350.sh
ts() { date '+%F %T'; }
PROD=$(docker ps --format '{{.Names}}' | grep -E '^ollama-0\.40\.' | head -1)
prod_idle() { [ "$(docker logs --since 15m "$PROD" 2>&1 | grep -cE '"/api/(generate|chat)"|"/v1/chat')" = 0 ] \
  && [ "$(curl -s --max-time 5 http://127.0.0.1:11497/api/ps | python3 -c 'import json,sys; print(len(json.load(sys.stdin).get("models",[])))' 2>/dev/null)" = 0 ]; }
wait_prod() { local n=0; until prod_idle; do [ $n -ge 36 ] && { echo "### production not idle for 3 h; going on GPU0's fit check alone $(ts)"; return; }
  [ $((n % 6)) -eq 0 ] && echo "### waiting for production to be idle: $1 $(ts)"; n=$((n+1)); sleep 300; done; echo "### production idle: $1 $(ts)"; }
cleanup() { docker logs "$CUR" > "$R/smoke-0405-runner.log" 2>&1; docker rm -f "$CUR" >/dev/null 2>&1; echo "### teardown $(ts) GPU0 $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0)"; }
echo "### SMOKE-0405 start $(ts)"
until grep -q 'PULL-0405-STOCK DONE' $S/pull-0405-stock.log 2>/dev/null; do sleep 120; done
grep -E 'ERROR' $S/pull-0405-stock.log && echo "### a pull reported an error (above)"
trap 'cleanup' EXIT
docker rm -f "$CUR" >/dev/null 2>&1
docker run -d --name "$CUR" --gpus all -v ollama_store_8tb:/root/.ollama -v "$S/jitcache:/jitcache" \
  -e CUDA_CACHE_PATH=/jitcache/nv -e MLX_PTX_CACHE_DIR=/jitcache/mlx-ptx-sm120-bebd6422 \
  -e OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0 -e OLLAMA_KV_CACHE_TYPE=f16 -e OLLAMA_FORMAT_TWO_PASS=1 \
  -p 127.0.0.1:$PORT:11434 -e OLLAMA_HOST=0.0.0.0:11434 -e OLLAMA_NOPRUNE=true -e OLLAMA_MAX_LOADED_MODELS=1 "$IMG" >/dev/null
v=""; for i in $(seq 1 90); do v=$(curl -sf --max-time 3 http://127.0.0.1:$PORT/api/version | sed -n 's/.*"version":"\([^"]*\)".*/\1/p'); [ -n "$v" ] && break; sleep 2; done
[ "$v" = "$WANT" ] || { echo "### smoke container serves '${v:-nothing}', not $WANT"; exit 2; }
curl -s http://127.0.0.1:$PORT/api/tags | python3 -c 'import json,sys; [print("### TAG",m["name"],m["digest"][:12],round(m["size"]/1e9,1),"GB",m.get("details",{}).get("format"),m.get("details",{}).get("quantization_level")) for m in json.load(sys.stdin)["models"] if "nemotron-3.5" in m["name"]]'
declare -A NEED=([nemotron-3.5-bolt:74b-a9b]=45000 [nemotron-3.5-super:120b-a12b]=88000)
for m in nemotron-3.5-bolt:74b-a9b nemotron-3.5-super:120b-a12b; do
  wait_prod "smoke $m"; wait_fit "${NEED[$m]}" "smoke $m"
  echo "### SMOKE $m $(ts)"
  python3 $S/smoke0405.py "$PORT" "$m" "$IMGF" "$R/smoke-0405-${m//[:\/]/_}.json"
  echo "### SMOKE $m rc=$? $(ts) launch: $(docker logs $CUR 2>&1 | grep -oE 'starting (llama-server|mlx runner).{0,200}' | tail -1 | cut -c1-200)"
  echo "### SMOKE $m loaded: $(docker logs $CUR 2>&1 | grep -iE 'offloaded [0-9]+/[0-9]+ layers|model weights|error|panic' | tail -3 | tr '\n' ' ' | cut -c1-400)"
  for x in $(curl -s --max-time 5 http://127.0.0.1:$PORT/api/ps | python3 -c 'import json,sys; print(" ".join(m["name"] for m in json.load(sys.stdin).get("models",[])))'); do
    curl -s --max-time 120 http://127.0.0.1:$PORT/api/generate -d "{\"model\":\"$x\",\"keep_alive\":0}" >/dev/null; done
done
cleanup; trap - EXIT
echo "### SMOKE-0405 DONE $(ts); relaunching loop908-0405.sh"
mv $S/loop908-0405.log $S/loop908-0405-held.log 2>/dev/null
setsid bash $S/loop908-0405.sh > $S/loop908-0405.log 2>&1 < /dev/null &
echo "### loop908 relaunched pid $!"
