#!/bin/sh
# Restart the vision-suite server for a benchmark run on APPLE SILICON, serving
# the MLX safetensors store. Use as RESTART_CMD.
#
# APPLE SILICON ONLY — and "MLX" alone does not mean Apple. The fork also ships
# an mlx_cuda_v13 payload for Linux/CUDA (see
# ../upstream-mlx-cuda-payload-unloadable.md), so a script called serve-apple-mlx.sh
# would read as if it applied there too. It does not: this one assumes a Metal
# host, a native (non-container) serve, and ~/.ollama/models-mlx. For CUDA or
# ROCm, restart the container instead — run_grid.sh's RESTART_CMD example shows
# the docker form.
#
#   RESTART_CMD='sh docs/maxusai/vision-suite/serve-apple-mlx.sh' \
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
# BIND defaults to 0.0.0.0 so remote agents and other machines on the network
# can drive a benchmark host. WHAT THAT MEANS, stated once so it is a choice
# rather than a surprise: ollama has no authentication, so every interface this
# binds reaches /api/generate, /api/pull, /api/delete and the whole model store
# with no credential. Narrow it with OLLAMA_BIND=127.0.0.1 (loopback only) or to
# a specific interface address; that is the right setting on any host whose
# network you do not control.
#
# It was 127.0.0.1 until 2026-08-17, which is why a remote agent could not reach
# :11436 while the server was plainly up and answering locally.
#
# Override any of these from the environment.
set -u
BIND="${OLLAMA_BIND:-0.0.0.0}"
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
OLLAMA_HOST="$BIND:$PORT" OLLAMA_MODELS="$STORE" \
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
