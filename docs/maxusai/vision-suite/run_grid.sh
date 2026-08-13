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
# Per-mode, PER-MODEL allowances. A single num_predict/num_ctx across models is
# wrong in both directions — it wastes context on models that terminate early and
# truncates the ones that do not. Values live in model_limits.env, derived from the
# maximum observed VALID prompt+eval per model; see that file for provenance and
# for why qwen3.6's think-on cells are unmeasured rather than tuned.
#
# The window must rise with the cap: the server rejects a request whose
# prompt + num_predict exceeds num_ctx, so raising num_predict alone turns a silent
# truncation into a hard 400. vision_suite.py reports that case with the required
# number rather than a bare "HTTP Error 400".
. "$(dirname "$0")/model_limits.env"

# limits_for <model> <think> -> "num_predict num_ctx"
limits_for() {
  _m=$1; _t=$2
  if [ "$_t" != "on" ]; then echo "$VS_NUM_PREDICT_OFF $VS_NUM_CTX_OFF"; return; fi
  # key on the model family, not the full tag, so quant/size variants inherit
  case "$_m" in
    gemma4*)    _v=$VS_THINKON_gemma4 ;;
    nemotron3*) _v=$VS_THINKON_nemotron3 ;;
    qwen3.6*|qwen3_6*) _v=$VS_THINKON_qwen3_6 ;;
    *)          _v=$VS_THINKON_DEFAULT ;;
  esac
  echo "${_v%%:*} ${_v##*:}"
}

for model in $MODELS; do
  for think in $THINK_MODES; do
    if [ -n "${RESTART_CMD:-}" ]; then
      echo ">>> $RESTART_CMD"
      $RESTART_CMD >/dev/null
      sleep 6
    fi
    slug="${PREFIX}-$(echo "$model" | tr ':/.' '---')-think${think}"
    set -- $(limits_for "$model" "$think")
    np="${NUM_PREDICT:-$1}"; nc="${NUM_CTX:-$2}"
    echo "########## $model think:$think -> $slug (num_predict=$np num_ctx=$nc) ##########"
    date +%H:%M:%S
    THINK="$think" NUM_PREDICT="$np" NUM_CTX="$nc" \
      python3 vision_suite.py "$HOST" "$slug" "$model"
  done
done
echo "GRID DONE ($PREFIX)"
