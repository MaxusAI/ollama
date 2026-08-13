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
# ENDPOINT/THINK/NUM_PREDICT/... pass through to both harnesses (defaults:
# chat endpoint, think off — the 2026-08-08 MLX-vs-GGUF campaign settings).
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

# Both think modes, always. The README's Method section has always said "always run
# both think modes", and this runner did not: it took a single THINK (default false)
# and ran one pass per model. That is not a stylistic gap — think-on is where
# multi_3img fails on nemotron3 and qwen3.6 while gemma4 passes, and where gemma4's
# document name_bbox IoU *improves* (0.712 -> 0.771). A think-off-only campaign
# reports none of it.
#
# NUM_PREDICT is per-mode because one cap cannot serve both. Think-off completes in
# a few hundred tokens; think-on needs far more, and a too-low cap does not look
# like truncation — the whole allowance is spent inside an unclosed thinking block
# and `response` comes back EMPTY, which reads as a vision failure. Measured:
# gemma4 scene_single needed 9,004 tokens; at 4,000 it returned nothing.
#
# NUM_CTX MUST rise with it. The server rejects a request whose prompt + num_predict
# exceeds the context window, so raising only the cap converts a silent truncation
# into a hard 400:
#   request (17574 tokens) exceeds the available context size (16384 tokens)
# 16384 is not enough for think-on: the worst observed prompt is multi_3img at
# 6,202 tokens (nemotron3, 3 images), so the floor is 6202 + 16000 = 22,202.
# 24576 leaves headroom for a larger prompt without another round of 400s.
# Override NUM_PREDICT / NUM_CTX to pin both modes to one value (not recommended).
THINK_MODES="${THINK_MODES:-false on}"

for m in $MODELS; do
  base_tag=$(printf '%s' "$m" | tr ':.' '__')
  for think in $THINK_MODES; do
    tag="${base_tag}-think${think}"
    if [ "$think" = "on" ]; then
      np="${NUM_PREDICT:-16000}"; nc="${NUM_CTX:-24576}"
    else
      np="${NUM_PREDICT:-4000}";  nc="${NUM_CTX:-16384}"
    fi
    pmode=$( (pmset -g 2>/dev/null || true) | awk '/powermode/{print $2}')
    echo "##### MODEL $m think=$think tag=$tag $(date +%T) powermode=${pmode:-n/a} num_predict=$np num_ctx=$nc"
    if [ -n "${RESTART_CMD:-}" ]; then
      sh -c "$RESTART_CMD"
      i=0
      until curl -sf "$HOST/api/version" >/dev/null 2>&1; do
        i=$((i + 1))
        [ "$i" -ge 60 ] && { echo "SERVER FAILED TO START for $m"; exit 1; }
        sleep 1
      done
    fi
    # vision_suite.py runs the fine-text probe itself since the fold (1db8ec9c), so
    # the separate finetext_probe.py call that used to live here is gone — it was a
    # second model load and a second transcription pass for a result already in
    # scores_<tag>.json, with nothing checking the two agreed.
    ENDPOINT="${ENDPOINT:-chat}" THINK="$think" NUM_PREDICT="$np" NUM_CTX="$nc" \
      python3 "$DIR/vision_suite.py" "$HOST" "$tag" "$m"
    echo "##### DONE $m think=$think $(date +%T)"
  done
done
echo "ENGINE COMPARE DONE"
