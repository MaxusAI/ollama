# ADR 0014: Media prompts prefill in span-aligned chunks, and refuse a late image

- **Status:** accepted, implemented 2026-08-10 on fork `main` lineage. The
  complete fix is deferred — see Consequences.
- **Date:** 2026-08-10
- **Deciders:** MaxusAI fork maintainers

## Context

`x/mlxrunner/pipeline.go` forced `prefillChunk = len(inputs)` for any vision
request, because a bidirectional image block must never straddle a chunk
boundary. gemma4 then builds a **dense** `L×K` float32 additive mask over
that chunk (`visionChunkMask`), once per attention window, and both are
retained in `b.Memo` for the whole forward — 8 bytes per cell.

With the chunk spanning the entire prompt, that mask is quadratic in the
prompt rather than in the image blocks. Nothing caps it: the MLX client's
`num_ctx` is a reported soft limit, server-side truncation is disabled for
MLX, and the model's `MaxPositionEmbeddings` is 131072. One image plus a
~32k-token prompt allocates on the order of gigabytes before a token is
produced.

Two facts in the model layer constrain any fix: `x/models/gemma4/gemma4.go`
gates the bidirectional path on `SeqOffsets[0] == 0` and, in that branch,
treats the chunk's own k/v as the complete key set; and `visionChunkMask`
indexes the key axis as **absolute** prompt positions while `K` is only the
chunk length. Together these mean a chunk carrying an image block must start
at position zero. Putting a block in a mid-prompt chunk would not crash — it
would silently fall through to the plain causal path and the image tokens
would lose bidirectional attention entirely.

## Decision

Two changes, both in `x/mlxrunner/pipeline.go`:

- Media prompts are chunked again, with boundaries snapped clear of any
  block's interior (`prefillChunkLen`). The opening chunk grows to cover
  every image block; everything after the last block is chunked normally.
  The invariant the original comment protected — a block is always processed
  in one pass, from position zero — is preserved and asserted by
  `TestPrefillChunkLen`.
- `Prepare` refuses a prompt whose last image ends so late that the opening
  chunk's mask would exceed `visionPrefillMaskBudget` (1 GiB, i.e. ~11.6k
  tokens before the last image's end), naming the limit and the remedy
  (`checkVisionPrefillBudget`, `TestCheckVisionPrefillBudget`).

## Options considered

- **Span-aligned chunking + an admission ceiling** (chosen) — removes the
  cost for the common "image early, long answer" shape and converts the
  remaining exposure from an OOM into a clean, actionable error. Both changes
  sit in the runner, where the span layout is already known.
- **Make `visionChunkMask` offset-aware and teach the bidi branch to attend
  over cache history** — the complete fix, and the one that would let a block
  ride a mid-prompt chunk. Deferred, not rejected: it is a model-layer change
  whose only real validation is the golden-parity and end-to-end vision
  tests, which are env-gated (`OLLAMA_VISION_E2E=1`) and need real weights on
  an Apple Silicon host. Getting it wrong degrades vision quality *silently*
  rather than failing a test, so it should not be attempted blind.
- **Leave media prompts un-chunked** — status quo; a remote unauthenticated
  request OOMs the runner.
- **Silently truncate the prompt** — drops user content without saying so;
  the fork rejects rather than degrades.

## Consequences

- The DoS is closed at the admission boundary on every prompt shape, and the
  mask cost is genuinely removed for images early in the prompt (image at
  token ~1k of 32k: ~13 MB instead of ~8.6 GB).
- **Negative:** "long text, then an image at the very end" is now refused
  with a 400 where it previously attempted the request and, on a
  large-memory host, might have completed. This is the deliberate trade —
  a named error over an OOM that takes the model offline.
- **Follow-up (required to lift the ceiling):** make `visionChunkMask`
  offset-aware and the bidirectional branch cache-history-aware, verify with
  the golden-parity and e2e vision suites enabled on the Apple Silicon host,
  then raise or remove `visionPrefillMaskBudget`. Until then the constant is
  the knob.
