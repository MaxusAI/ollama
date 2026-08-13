#!/usr/bin/env bash
# Model × think-mode grid runner for vision_suite.py.
#
#   RESTART_CMD="docker restart <container>" \
#   MODELS="nemotron3:33b-q4_K_M gemma4:31b-it-q4_K_M" \
#   THINK_MODES="false on" NUM_PREDICT=4000 \
#   ./run_grid.sh <host> <tag-prefix>
#
# One vision_suite.py run per (model, think) cell, cold-restarting the server
# between cells when RESTART_CMD is set (required on payloads with
# cross-request leakage — upstream #17475).
set -u
cd "$(dirname "$0")"
HOST="${1:?usage: run_grid.sh <host> <tag-prefix>}"
PREFIX="${2:?usage: run_grid.sh <host> <tag-prefix>}"
MODELS="${MODELS:-nemotron3:33b-q4_K_M gemma4:31b-it-q4_K_M qwen3.6:35b-a3b-q4_k_m}"
THINK_MODES="${THINK_MODES:-false on}"
# Per-mode allowances. A single NUM_PREDICT cannot serve both modes, and the
# context window must rise with the cap: the server rejects a request whose
# prompt + num_predict exceeds num_ctx, so raising only NUM_PREDICT turns a
# silent truncation into a hard 400 —
#   request (17574 tokens) exceeds the available context size (16384 tokens)
# which is exactly what this runner produced with NUM_PREDICT=16000 at the
# default 16384 window. Worst observed think-on prompt is multi_3img at 6,202
# tokens, so the think-on floor is 6202 + 16000 = 22,202; 24576 leaves headroom.
# Setting NUM_PREDICT/NUM_CTX in the environment pins both modes to one value.
NUM_PREDICT_OFF="${NUM_PREDICT:-4000}"
NUM_PREDICT_ON="${NUM_PREDICT:-16000}"
NUM_CTX_OFF="${NUM_CTX:-16384}"
NUM_CTX_ON="${NUM_CTX:-24576}"

for model in $MODELS; do
  for think in $THINK_MODES; do
    if [ -n "${RESTART_CMD:-}" ]; then
      echo ">>> $RESTART_CMD"
      $RESTART_CMD >/dev/null
      sleep 6
    fi
    slug="${PREFIX}-$(echo "$model" | tr ':/.' '---')-think${think}"
    if [ "$think" = "on" ]; then np="$NUM_PREDICT_ON"; nc="$NUM_CTX_ON"
    else np="$NUM_PREDICT_OFF"; nc="$NUM_CTX_OFF"; fi
    echo "########## $model think:$think -> $slug (num_predict=$np num_ctx=$nc) ##########"
    date +%H:%M:%S
    THINK="$think" NUM_PREDICT="$np" NUM_CTX="$nc" \
      python3 vision_suite.py "$HOST" "$slug" "$model"
  done
done
echo "GRID DONE ($PREFIX)"
