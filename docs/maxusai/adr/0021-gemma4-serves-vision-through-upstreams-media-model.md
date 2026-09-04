# ADR 0021: gemma4 serves vision through upstream's `base.MediaModel`, with a fork-local budget seam

- **Status:** accepted (2026-08-11). Upheld by decision D1-A of the v0.33.3
  fold (2026-09-04): upstream shipped its own gemma4 MLX vision and audio,
  and this fork keeps the tower below because upstream's has no per-request
  budget seam. Two mechanisms named here have moved and the decisions have
  not: decision 6's test seam is now `mlxtest.Run` / `RunSubtest` on
  `x/internal/mlxthreadtest` (`mlx.ClaimOSThread` was deleted with upstream's
  MLX-C error-contract rewrite — see ADR 0017's status), and the same applies
  to the `mlxtest.Setup` divergence listed under Consequences.
- **Date:** 2026-08-11
- **Deciders:** MaxusAI fork maintainers
- **Related:** ADR 0020 (`0020-mlx-vision-lineage-preserves-the-pre-upstream-fork.md`, on `release/mlx-vision` — not present on `main`)
  (the lineage this unblocks), [ADR 0014](0014-media-prompts-chunk-around-image-blocks.md)
  (chunking and the admission ceiling), [ADR 0003](0003-vision-image-token-budget-policy.md) /
  [0007](0007-gemma4-default-budget-560.md) / [0008](0008-gemma4-budget-fill-restores-1120.md)
  (the image-token budget), [ADR 0009](0009-mlx-pure-go-constrained-sampling.md)
  (no silent drops), [ADR 0017](0017-mlx-work-runs-on-a-permanently-claimed-os-thread.md)
  (MLX thread affinity)

## Context

Upstream reworked MLX media handling around a new `base.MediaModel`
(`PrepareMedia` / `EncodeMedia`), and the runner now gates every media request on
it. Upstream's `glimmer` and `qwen3_5` implement it; gemma4 implemented the fork's
`base.VisionModel` instead. Taking upstream therefore turned the fork's entire
vision line off — compiling perfectly, with the acceptance suites gated behind
`OLLAMA_VISION_E2E=1`, so CI would have stayed green while the feature was dead.

That blocked `main` from following upstream at all, which is why ADR 0020 exists.
This ADR records the port that unblocks it.

Two constraints shaped the work, neither obvious from upstream's interface:

- **No upstream model exercises the non-causal path.** `glimmer` and `qwen3_5`
  both set `Causal: true`. gemma4's image spans attend bidirectionally, so it is
  the first consumer of `PreparedItem.Causal == false`.
- **The image-token budget is fork-only.** `ImageMinTokens`/`ImageMaxTokens` do
  not exist upstream, and `PrepareMedia` takes no options, so the per-request
  budget had nowhere to arrive.

## Decision

1. **gemma4 implements `base.MediaModel`.** `PrepareMediaWithBudget` splices
   BOI + one placeholder per soft token + EOI, marks the item non-causal, and
   reuses `NewVisionInput`'s preprocessing verbatim so ADR 0015's
   alpha-over-white compositing and ADR 0008's budget-fill sizing are shared
   rather than reimplemented. `EncodeMedia` runs the tower; `scatterMedia`
   replaces `MergedEmbeddings`, splicing features over the placeholder rows.
2. **A fork-local `base.MediaBudgetModel` carries the budget**, rather than
   changing upstream's `MediaModel`. A second interface means a future merge
   conflicts on one added block instead of on every media model.
3. **Every MLX media model implements it** — `qwen3_5` and `glimmer` too — so
   `image_max_tokens` is never silently ignored (ADR 0009). Each keeps its own
   ceiling: a value equal to the shared api default counts as unset, the
   convention `llm/llama_server.go` already applies for nemotron and qwen-VL.
4. **Bidirectional spans come from `b.Media`, and the opening chunk carries every
   one of them.** The `SeqOffsets[0] == 0` requirement is kept, because it is
   load-bearing rather than vestigial: the bidi path attends over the chunk's own
   k/v rather than the cache history — routing through history lets the sliding
   applier re-add the window over relaxed blocks — so the mask's key axis is the
   chunk's keys, and those are the complete key set only at offset zero.
   Upstream's `extendChunk` does not preserve that: it prevents a chunk *ending*
   inside an atomic expansion but explicitly permits resuming inside one. The
   fork-local `requestMedia.growOpeningChunk` re-establishes it, kept separate
   from `extendChunk` so upstream's rule keeps its exact semantics and tests.
5. **ADR 0014's admission ceiling stays**, keyed on media items rather than the
   removed `VisionSpans`.
6. **The test seam converges on upstream's `x/internal/mlxtest`**, implemented
   with `mlx.ClaimOSThread()` and no unlock (ADR 0017/0018).

## Alternatives considered

- **Keep `base.VisionModel` alongside upstream's path.** Rejected: the runner
  gates on `MediaModel`, so the fork's interface would have had no caller. Two
  media paths in one runner is also exactly the divergence that made this merge
  expensive.
- **Change `MediaModel` to take the budget.** Fewer types, but it conflicts on
  every model file on every merge, and forces upstream models to carry a
  parameter they ignore.
- **Reject requests carrying a budget on budget-blind models.** Honest, and
  unusable: `api.DefaultOptions` always populates the fields, so the gate would
  have refused all qwen and glimmer image traffic.
- **Drop ADR 0014's ceiling as subsumed by `extendChunk`.** Measured and refuted;
  see Consequences.

## Consequences

- Positive: `main` can follow upstream without losing vision. gemma4 answers
  image requests through the same seam upstream's own models use.
- Positive: the budget reaches preprocessing per request on **all three** MLX
  media models, and `PreparedItem.Dims` moves with it, so two budgets over
  identical bytes cannot share a prefix-cache prefix.
- Negative: three fork-local divergences to re-resolve on every upstream merge —
  `MediaBudgetModel` and `ResolveImageBudget` in `base`, the budget methods on
  `qwen3_5` and `glimmer`, and `mlxtest.Setup`'s permanent claim. Each is
  additive and small by design.
- Negative: gemma4 remains the only consumer of the non-causal media path, so
  upstream changes there will land untested against our only user of them.
- **ADR 0014's ceiling is not subsumed.** A non-causal expansion forces a dense
  `[chunkLen, keyCount]` overlay, and `keyCount` is the cache length, so the cost
  is chunk × context. At the 2 KiB prefill chunk that is 0.25 GiB at 32k and
  1.00 GiB at 128k, and `extendChunk` can grow the chunk to a whole expansion,
  reaching 2.00 GiB. The guard is retained and now charges only non-causal items.
- Follow-up: none blocking. `x/imagegen` is out of scope by ADR 0019/0020.

## Conformance

- Unit: the expansion is bracketed and non-causal; the same bytes at two budgets
  give different `Dims` and a longer expansion; an unset budget resolves the
  model's own default; a non-image segment is refused;
  `ResolveImageBudget` treats the shared default as unset.
- Mask: a query attends a later key inside its own span where a causal baseline
  forbids it. **Model output cannot gate this** — the end-to-end spatial probe
  passed with bidirectional attention entirely off, so the mask is compared
  directly.
- Chunking: the opening chunk grows past every bidirectional expansion, later
  chunks are untouched, causal media is not charged for it, and growth is clipped
  to keep the decode seed. This guards the precondition above, and it exists
  because an earlier revision removed the offset gate on the strength of a unit
  test that asserted the key axis was absolute — which is precisely what the test
  had assumed rather than checked. Multi-chunk media then produced garbage logits,
  surfacing as `constrained sampling produced an illegal token`. It was caught by
  the benchmark suite, not the test suite; the chunking test above is the guard
  that would have caught it.
- Ceiling: table cases plus a test that computes the allocation an admitted
  request would make and fails if anything admitted exceeds the budget.
- Hardware (`gemma4:12b-nvfp4`, Apple Silicon): golden parity against the
  vendored mlx-vlm reference — mean -0.02562 vs -0.02560, std 2.4605 vs 2.4613,
  norm_mean 151.310 vs 151.352, max sampled element delta 0.0625 against a 0.15
  bound — and both end-to-end subtests.
- Preflight (`--platform apple-silicon-mlx`, server `0.32.5-maxusai-aff5179f`):
  token ladder 5/5 within ±2 and text prefix 19 on both `gemma4` and
  `gemma4_unified`, matching the 2026-08-08 baseline; no contention.
