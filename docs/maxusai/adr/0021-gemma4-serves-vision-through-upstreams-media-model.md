# ADR 0021: gemma4 serves vision through upstream's `base.MediaModel`, with a fork-local budget seam

- **Status:** accepted (2026-08-11). Upheld by decision D1-A of the v0.33.3
  fold (2026-09-04): upstream shipped its own gemma4 MLX vision and audio,
  and this fork keeps the tower below because upstream's has no per-request
  budget seam. Two mechanisms named here have moved and the decisions have
  not: decision 6's test seam is now `mlxtest.Run` / `RunSubtest` on
  `mlx/mlxthread/mlxthreadtest` (`mlx.ClaimOSThread` was deleted with upstream's
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
6. **The test seam converges on upstream's `mlx/mlxtest`**, implemented
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

## Amendment 2026-09-24 — nemotron_h is the fourth media model

Upstream v0.34.3 (#17714) gave `nemotron_h` vision on MLX. It implements `model.MediaModel`
and not `model.MediaBudgetModel`, so after the fold `image_min_tokens`/`image_max_tokens`
reached the runner's fallback (`mlxrunner/media.go`: a warning, then the model's own bounds)
— the state Decision 3 exists to rule out, on the one architecture whose GGUF path honours
the budget through compat `002`. The v0.34.3 fold landed it that way and recorded it as an
open item; this amendment closes it.

**Semantics are the GGUF path's** (`nemotronImageTokenBudget`, `llm/llama_server.go`), so one
request means the same thing on both engines:

- an unset bound, or one equal to the shared api default, keeps the model's own —
  `min_num_patches` and the context-bound `nemotronImagePatchBudget`, kept in patch units so
  an unset request reproduces upstream's preprocessing exactly;
- one token is `DownsampleFactor²` patches (×4, the 2×2 shuffle);
- the maximum is clamped to the model's ceiling (13312 patches, 3328 tokens for Omni);
- the minimum is clamped down to the maximum, and **neither is floored at
  `min_num_patches`**: `image_max_tokens=128` gets about 128 tokens, as llama-server gives it.

**Placement follows glimmer's.** The resolution lives in a fork-local
`mlxrunner/model/nemotron_h/media_budget.go`, including the `MediaBudgetModel` assertion.
Upstream's `vision_prompt.go` changes in three places: `PrepareMedia` delegates to
`PrepareMediaWithBudget(segments, 0, 0)` with its body kept in place, the call site passes the
resolved bounds, and `preprocessImage` takes the minimum. The request's minimum reaches
upstream's `nemotronImagePatchGrid` through a copy of the config with `MinNumPatches`
replaced, so that function and the upstream tests that call it are untouched.

**Measured on the preprocessing, not yet on a GPU** (`media_budget_test.go`, pure Go,
Omni's c-radio_v4-h configuration; feature tokens, excluding the two markers):

| image | unset | `max=1024` | `max=128` | `min=1024` |
|---|---|---|---|---|
| 320×240 | 270 | 270 | 80 | 1044 |
| 640×480 | 300 | 300 | 117 | 1036 |
| 1920×1080 | 2040 | 1008 | 120 | 2040 |
| 2048×2048 | 3306 | 1024 | 121 | 3306 |

The unset column is the GGUF path's count for the same images
([nemotron-test-image.md](../nemotron-test-image.md): 640×480 → 302 and 1920×1080 → ≈2042 with
the markers; 320×240 inside the 258–320 floor band). Every budget that moves the grid moves
`Dims` and the expansion length, which is how it reaches cache identity; budgets that resolve to
the same grid produce the same pixels, and sharing their prefix is correct.

**Not measured here:** the end-to-end run on MLX hardware — a served
`image_max_tokens=1024` request's `prompt_eval_count`, and two budgets over one image missing
each other in the prefix cache. The gfx1151 host has no MLX runtime; the Apple Silicon or
CUDA host owns that check. Decision 3 holds again for all four media models, and the
Consequences' list of fork-local divergences gains the budget methods on `nemotron_h`.
