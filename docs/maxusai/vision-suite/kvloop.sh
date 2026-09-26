#!/usr/bin/env bash
# kvloop.sh -- KV cache type x flash attention against think-on loops, on a docker host (ROCm or CUDA).
#
# One container per arm, every case captured cold by thinkcap.py (the suite's own request, every model evicted
# first). The protocol and the results are in docs/maxusai/tasks/kv-precision-think-loops.md (ADR 0043).
#
# Required:
#   IMG        the image under test
#   CASES      space-separated <model>|<test> pairs, e.g. "gemma4:26b-a4b-it-q4_K_M|bbox_contract_real_1img"
#   STORE      the model store to bind-mount at /root/.ollama
#   GPU_ARGS   device flags: ROCm "--device /dev/kfd --device /dev/dri --group-add <video gid> --group-add
#              <render gid>", CUDA "--gpus all"
# Optional:
#   ARMS       space-separated <kv>:<fa> pairs, fa 1 = flash attention on, 0 = off
#              (default "f16:1 f16:0 f32:0 f32:1"; q8_0:0 does not exist, a quantized V cache needs FA)
#   NUM_CTX    capture context, default 65536 (num_predict = NUM_CTX - 8192)
#   OUT        output directory, default ./kvloop-out
#   PORT       host port, default 11494;  NAME  container name, default ollama-kvloop
#   EXTRA_ENV  more "-e K=V" flags, to match production's environment
#
# OLLAMA_FLASH_ATTENTION=0 makes the fork pass --flash-attn off; unset would be "auto", which enables it.
# Each capture logs the runner's --cache-type-k, --cache-type-v and --flash-attn flags, so the arm is proven,
# not assumed. Greedy results do not move with GPU sharing; timing does.
set -uo pipefail
: "${IMG:?set IMG}" "${CASES:?set CASES}" "${STORE:?set STORE}" "${GPU_ARGS:?set GPU_ARGS}"
ARMS=${ARMS:-"f16:1 f16:0 f32:0 f32:1"}
NUM_CTX=${NUM_CTX:-65536}; OUT=${OUT:-./kvloop-out}; PORT=${PORT:-11494}; NAME=${NAME:-ollama-kvloop}
EXTRA_ENV=${EXTRA_ENV:-}
VS=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
export HTTP_TIMEOUT=${HTTP_TIMEOUT:-9000}   # a looping capture at shared-GPU speed outlives the derived timeout

start() {  # start <kv> <fa>
  docker rm -f "$NAME" >/dev/null 2>&1
  # shellcheck disable=SC2086  # GPU_ARGS and EXTRA_ENV are flag lists
  docker run -d --name "$NAME" -p "$PORT:11434" $GPU_ARGS --ipc host --shm-size 16g \
    -v "$STORE:/root/.ollama" -e OLLAMA_HOST=0.0.0.0:11434 -e OLLAMA_MODELS=/root/.ollama/models \
    -e OLLAMA_DEBUG=1 -e OLLAMA_NUM_PARALLEL=2 -e OLLAMA_MAX_LOADED_MODELS=1 -e OLLAMA_NOPRUNE=1 \
    -e OLLAMA_KV_CACHE_TYPE="$1" -e OLLAMA_FLASH_ATTENTION="$2" $EXTRA_ENV "$IMG" >/dev/null || return 1
  until curl -sf "http://127.0.0.1:$PORT/api/version" >/dev/null; do sleep 1; done
  echo "== kv=$1 fa=$2 image=$IMG $(date -Is)"
}

for arm in $ARMS; do
  kv=${arm%%:*}; fa=${arm##*:}
  tag="$kv-fa$([ "$fa" = 1 ] && echo on || echo off)"
  start "$kv" "$fa" || { echo "could not start $tag"; continue; }
  for c in $CASES; do
    model=${c%%|*}; test=${c##*|}
    f="$OUT/${tag}_$(echo "$model" | tr ':/' '__')_${test}_${NUM_CTX}.json"
    if [ -s "$f" ]; then echo "have $f"; continue; fi
    echo "-- $tag $model $test $NUM_CTX $(date -Is)"
    (cd "$VS" && timeout $((HTTP_TIMEOUT + 500)) python3 thinkcap.py "http://127.0.0.1:$PORT" "$model" "$test" "$NUM_CTX" "$f") \
      || echo "capture $tag $model $test exited $?"
    echo "   runner flags: $(docker logs "$NAME" 2>&1 | command grep -a 'starting llama-server' | tail -1 \
      | command grep -oE -- '--(cache-type-[kv]|flash-attn) [a-z0-9_]+' | tr '\n' ' ')"
  done
  docker logs "$NAME" > "$OUT/server-$tag.log" 2>&1
done
docker rm -f "$NAME" >/dev/null 2>&1
echo "== kvloop done $(date -Is)"
