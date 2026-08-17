#!/bin/sh
# Restart the vision-suite server for a benchmark run. Use as RESTART_CMD.
#
#   RESTART_CMD='sh docs/maxusai/vision-suite/serve-mlx.sh' \
#     MODELS="..." ./run_engine_compare.sh http://127.0.0.1:11436
#
# THIS SCRIPT IS IN THE REPO ON PURPOSE. It used to live in /tmp, which meant
# the one line that keeps a sweep from exhausting the host was lost on reboot
# and invisible to everyone else. An operational guard that is not version
# controlled is not a guard.
#
# OLLAMA_MAX_LOADED_MODELS=1 IS THE POINT OF THIS FILE. The scheduler keeps
# every model it has served resident, so a sweep over N models holds N at once —
# COUNT, not size, is what exhausts the machine. Measured 2026-08-17 on a 128 GB
# M-series Mac: a four-arch preflight ladder left three runners resident at
# 68.7 + 22.0 + 20.7 GB, reaching 106 GB used and 53.9 GB of swap while Docker's
# VM (19.9 GB) and two other VMs were live — one allocation from OOM-killing
# unrelated work. The cap costs a reload between models, which every runner here
# pays anyway for cold-cache reasons.
#
# Override any of these from the environment.
set -u
BIN="${OLLAMA_BIN:-/tmp/ollama-vs}"
PORT="${OLLAMA_PORT:-11436}"
STORE="${OLLAMA_STORE:-$HOME/.ollama/models-mlx}"
MAX_LOADED="${OLLAMA_MAX_LOADED_MODELS:-1}"
LOG="${OLLAMA_SERVE_LOG:-/tmp/vs.log}"
REPO="${OLLAMA_REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"

[ -x "$BIN" ] || { echo "serve-mlx: no executable at $BIN (set OLLAMA_BIN)" >&2; exit 1; }

pkill -f "$BIN serve" 2>/dev/null
sleep 3
# Start from the repo root: llama-server is resolved relative to the working
# directory, and starting elsewhere fails with "llama-server binary not found".
cd "$REPO" || exit 1
OLLAMA_HOST="127.0.0.1:$PORT" OLLAMA_MODELS="$STORE" \
  OLLAMA_MAX_LOADED_MODELS="$MAX_LOADED" \
  "$BIN" serve > "$LOG" 2>&1 &

i=0
while [ "$i" -lt 60 ]; do
  if curl -s --max-time 2 "http://127.0.0.1:$PORT/api/version" >/dev/null 2>&1; then
    exit 0
  fi
  i=$((i + 1))
  sleep 2
done
echo "serve-mlx: server did not answer on :$PORT within 120s; see $LOG" >&2
exit 1
