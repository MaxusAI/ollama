#!/usr/bin/env bash
# Model × think-mode grid runner for vision_suite.py.
#
#   RESTART_CMD="docker restart <container>" \
#   MODELS="nemotron3:33b-q4_K_M gemma4:31b-it-q4_K_M" \
#   THINK_MODES="false on" \
#   ./run_grid.sh <host> <tag-prefix>
#
# One vision_suite.py run per (model, think) cell, cold-restarting the server
# between cells when RESTART_CMD is set (required on payloads with
# cross-request leakage — upstream #17475).
#
# BUDGETS ARE PER MODE, and num_ctx rises with num_predict. One NUM_PREDICT
# cannot serve both: a reasoning model spends its first tokens thinking, so a
# think-off-sized cap returns an EMPTY response with eval_count == num_predict —
# indistinguishable from a vision failure. Raising the cap alone then converts
# that truncation into a hard 400, because the server checks
# prompt + num_predict <= num_ctx up front rather than truncating:
#   request (17574 tokens) exceeds the available context size (16384 tokens)
# Same rule and same reserve as run_engine_compare.sh — see its header for the
# measurements. Setting NUM_PREDICT or NUM_CTX pins BOTH modes to one value;
# use NUM_PREDICT_THINKON / NUM_CTX_THINKON to move think-on alone.
#
# CLIMB THE LADDER; DO NOT START AT THE TOP. Think-on starts at the 16384 rung
# with num_predict derived as (num_ctx - reserve). Unlike run_engine_compare.sh
# this runner does NOT auto-escalate — it reports a capped cell and the rung to
# retry at, so a grid stays one cell per (model, think).
set -u

# ENDPOINT pinned to generate. The suite default flipped to chat on
# 2026-08-19; every tag this runner has ever produced was measured on
# /api/generate, and the two endpoints are only MEASURED equivalent on one
# model and one build (see client.endpoint()), not guaranteed. Pinning keeps
# old and new tags comparable; change it deliberately, not by inheriting a
# default that moved underneath this file.
ENDPOINT="${ENDPOINT:-generate}"
export ENDPOINT

cd "$(dirname "$0")"
HOST="${1:?usage: run_grid.sh <host> <tag-prefix>}"
PREFIX="${2:?usage: run_grid.sh <host> <tag-prefix>}"
MODELS="${MODELS:-nemotron3:33b-q4_K_M gemma4:31b-it-q4_K_M qwen3.6:35b-a3b-q4_k_m}"
THINK_MODES="${THINK_MODES:-false on}"
# Must exceed the largest prompt the suite produces: nemotron3 multi_3img at
# 6,203 tokens is the worst case (gemma4 is only 3,765, which is why a 4096
# reserve appeared to work until nemotron was run).
CTX_PROMPT_RESERVE="${CTX_PROMPT_RESERVE:-8192}"

for model in $MODELS; do
  for think in $THINK_MODES; do
    if [ -n "${RESTART_CMD:-}" ]; then
      echo ">>> $RESTART_CMD"
      $RESTART_CMD >/dev/null
      sleep 6
    fi
    slug="${PREFIX}-$(echo "$model" | tr ':/.' '---')-think${think}"
    if [ "$think" = "on" ]; then
      nc="${NUM_CTX:-${NUM_CTX_THINKON:-16384}}"
      np="${NUM_PREDICT:-${NUM_PREDICT_THINKON:-$((nc - CTX_PROMPT_RESERVE))}}"
    else
      nc="${NUM_CTX:-16384}"
      np="${NUM_PREDICT:-4000}"
    fi
    if [ "$((np + CTX_PROMPT_RESERVE))" -gt "$nc" ]; then
      echo "WARNING: num_predict=$np leaves <$CTX_PROMPT_RESERVE of num_ctx=$nc for the prompt;" \
           "the effective cap will be (num_ctx - prompt), not num_predict."
    fi
    echo "########## $model think:$think -> $slug (num_predict=$np num_ctx=$nc) ##########"
    date +%H:%M:%S
    THINK="$think" NUM_PREDICT="$np" NUM_CTX="$nc" \
      python3 vision_suite.py "$HOST" "$slug" "$model"

    # A cell at eval_count == num_predict is CAPPED, not a model failure — the
    # answer is truncated inside an unclosed thinking block and scores 0. Say so
    # here; the alternative is a zero that reads as a regression (ADR 0022).
    python3 - "scores_${slug}.json" "$np" "$nc" <<'PYEOF'
import json, sys
path, np, nc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    scores = json.load(open(path))
except (OSError, ValueError):
    sys.exit(0)
capped = [k for k, v in scores.items()
          if isinstance(v, dict) and v.get("eval_count") == np]
if capped:
    print(f"WARNING: capped at num_predict={np}: {', '.join(sorted(capped))}. "
          f"These scores are truncation, NOT quality. Re-run the cell with "
          f"NUM_CTX_THINKON={nc * 2}.")
PYEOF
  done
done
echo "GRID DONE ($PREFIX)"
