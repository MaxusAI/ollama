#!/bin/sh
# Engine-parity campaign: cold server per model, the three-suite run plus the
# fine-text probe, one tag per model. Pair with summarize_engine_compare.py,
# which renders the two comparison tables from the per-tag score files.
#
# Usage:
#   RESTART_CMD='<restart the serving process>' \
#   MODELS="gemma4:12b-nvfp4 gemma4:12b-it-q4_K_M ..." \
#     ./run_engine_compare.sh http://127.0.0.1:11499
#
# RESTART_CMD is the cold-server hook (see README "Method"); without it the
# models share one server process and cross-request leakage caveats apply.
# ENDPOINT/NUM_PREDICT/... pass through to both harnesses (default: chat
# endpoint — the 2026-08-08 MLX-vs-GGUF campaign settings).
#
# THINK MODES. Both are run per model, per the README's "always run both think
# modes": THINK_MODES defaults to "false on". Tags carry the mode
# (<model>_thinkfalse / <model>_thinkon) so the two cells cannot overwrite each
# other; summarize_engine_compare.py reads --think to pick which set to render
# and falls back to the bare <model> tag for pre-2026-08-09 runs.
#
# num_predict is raised for think-on. A reasoning model spends its first tokens
# thinking, and a cap below that budget returns an EMPTY response with a
# non-zero eval_count — indistinguishable from a vision failure, and the origin
# of this fork's "serves blank images" false alarm. Defaults are 2200 think-off
# and 16000 think-on; override with NUM_PREDICT (both modes) or
# NUM_PREDICT_THINKON (think-on only).
#
# Why 16000 and not the README's ">=4000 with THINK=on": measured 2026-08-09 on
# gemma4:12b, that floor does not hold. Every empty cell in that run sat at
# exactly eval=4000 while every populated one finished under it — MLX document
# 2798 and multi 3803, GGUF nothing at all (it decodes ~30 tok/s vs MLX's ~100,
# so it exhausted the budget on all three tests). 16000 is the README's own
# think-on multi-image figure, applied to every test.
#
# A cell that still reports eval == num_predict is capped, NOT a model failure.
# Check that before recording an empty result.
#
# NUM_CTX MUST HOLD prompt + num_predict. They share one window, so a large
# num_predict against a small num_ctx is unreachable — generation stops at the
# context limit, not the cap, and the effective budget is silently
# (num_ctx - prompt). Measured prompts here are 1687 scene / 1450 document /
# 3765 multi-image.
#
# CLIMB THE LADDER; DO NOT START AT THE TOP. num_ctx is 4096 / 8192 / 16384 /
# 32768 / 65536, and think-on starts at 16384 with num_predict = num_ctx - 4096
# (12288) — comfortably above the 4000 that demonstrably capped, while costing
# a quarter of the KV cache a blanket 32768 would. Every rung doubles KV, and
# 31B dense at 32768 is the cell that will hurt first.
#
# ESCALATE ONLY ON EVIDENCE. If a cell reports eval == num_predict it was
# capped; move that model up ONE rung and re-run just that cell:
#   NUM_CTX_THINKON=32768 MODELS="gemma4:31b-it-q4_K_M" THINK_MODES=on ./run_engine_compare.sh <host>
# num_predict follows num_ctx automatically, so the pair cannot go incoherent.
# Record the rung a result was measured at — cells from different rungs are not
# directly comparable on throughput.
#
# macOS + MLX note: a fork server binary must start with the repo root as its
# working directory (the MLX dylib and llama-server payload resolve relative
# to cwd/executable), e.g.
#   RESTART_CMD='pkill -f "ollama serve"; sleep 2; (cd /path/to/repo && \
#     OLLAMA_MODELS=$HOME/.ollama/models-mlx OLLAMA_HOST=127.0.0.1:11499 \
#     ./ollama serve >> /tmp/serve.log 2>&1 &)'
set -eu
HOST="${1:?usage: run_engine_compare.sh <host>}"
DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS="${MODELS:?set MODELS to the space-separated model list}"

THINK_MODES="${THINK_MODES:-false on}"

for m in $MODELS; do
  base=$(printf '%s' "$m" | tr ':.' '__')
  for think in $THINK_MODES; do
    tag="${base}_think${think}"
    # Reasoning models think before answering; too small a cap yields an empty
    # response, not a short one. See the header note.
    if [ "$think" = "on" ]; then
      # Ladder rung; num_predict is derived so the pair is always coherent.
      nc="${NUM_CTX:-${NUM_CTX_THINKON:-16384}}"
      np="${NUM_PREDICT:-${NUM_PREDICT_THINKON:-$((nc - 4096))}}"
    else
      nc="${NUM_CTX:-16384}"
      np="${NUM_PREDICT:-2200}"
    fi
    # prompt + num_predict share num_ctx; warn rather than silently under-run.
    if [ "$((np + 4096))" -gt "$nc" ]; then
      echo "WARNING: num_predict=$np leaves <4096 of num_ctx=$nc for the prompt;" \
           "the effective cap will be (num_ctx - prompt), not num_predict."
    fi
    pmode=$( (pmset -g 2>/dev/null || true) | awk '/powermode/{print $2}')
    echo "##### MODEL $m think=$think tag=$tag num_predict=$np num_ctx=$nc $(date +%T) powermode=${pmode:-n/a}"
    # Cold server per CELL, not per model: think-on and think-off are separate
    # cells and the leakage caveat applies between them too.
    if [ -n "${RESTART_CMD:-}" ]; then
      sh -c "$RESTART_CMD"
      i=0
      until curl -sf "$HOST/api/version" >/dev/null 2>&1; do
        i=$((i + 1))
        [ "$i" -ge 60 ] && { echo "SERVER FAILED TO START for $m think=$think"; exit 1; }
        sleep 1
      done
    fi
    ENDPOINT="${ENDPOINT:-chat}" THINK="$think" NUM_PREDICT="$np" NUM_CTX="$nc" \
      python3 "$DIR/vision_suite.py" "$HOST" "$tag" "$m"
    ENDPOINT="${ENDPOINT:-chat}" THINK="$think" NUM_PREDICT="$np" NUM_CTX="$nc" \
      python3 "$DIR/finetext_probe.py" "$HOST" "$tag" "$m"
    echo "##### DONE $m think=$think $(date +%T)"
  done
done
echo "ENGINE COMPARE DONE"
