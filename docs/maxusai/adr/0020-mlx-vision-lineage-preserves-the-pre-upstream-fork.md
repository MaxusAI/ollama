# ADR 0020: `release/mlx-vision` preserves the fork's complete pre-upstream state

- **Status:** accepted (2026-08-11)
- **Date:** 2026-08-11
- **Deciders:** MaxusAI fork maintainers
- **Related:** [ADR 0006](0006-release-lineage-is-never-merged-into-main.md) (lineages are
  never merged into `main`), ADR 0019 (on `release/imagegen-mlx`, which this supersedes),
  ADRs [0003](0003-vision-image-token-budget-policy.md) /
  [0014](0014-media-prompts-chunk-around-image-blocks.md) /
  [0015](0015-transparent-images-composite-over-white.md) /
  [0016](0016-reload-on-resolved-vision-flags.md) (the vision behaviour this protects)

> **This ADR lives only on `release/mlx-vision`.** `main` is returning to upstream, and this
> lineage is what `main` gives up to do it.

## Context

`main` is returning to following `upstream/main` (43 commits behind, 205 ahead). A trial merge
resolved every conflict and built cleanly, then revealed a blocker that no build or ungated test
catches:

**Upstream reworked media handling around a new `base.MediaModel` interface** (`PrepareMedia` /
`EncodeMedia`). The runner now gates every media request on `r.Model.(base.MediaModel)`. Only
upstream's `glimmer` and `qwen3_5` implement it. `gemma4` — the fork's only vision model —
implements the fork's `base.VisionModel` instead, so after the merge every image request against
gemma4 returns `this model does not support image input`, while `x/create` still advertises the
vision capability. The whole fork vision line goes dark **at runtime while compiling perfectly**,
and its acceptance suites are gated behind `OLLAMA_VISION_E2E=1` with real weights on Apple
Silicon, so CI stays green.

Collateral in the same merge: ADR 0014's 1 GiB dense-mask admission ceiling
(`checkVisionPrefillBudget`) has no upstream equivalent and was dropped, while the model-layer
precondition it protected (gemma4 gating bidirectional attention on `SeqOffsets[0] == 0`) is
unchanged; `batch.InputsEmbeds` and `BidiSpans` lose their only writer; and ADR 0014's two named
conformance tests no longer compile because the helpers they cover were deleted.

## Decision

`release/mlx-vision` is a maintained lineage at `a8a25886` — the last `main` before the upstream
merge — preserving the fork's complete pre-upstream state:

- the MLX vision line: gemma4's `base.VisionModel`, the budget ladder (ADR 0003/0007/0008),
  chunk-alignment and the admission ceiling (ADR 0014), alpha compositing (ADR 0015), and
  reload-on-resolved-flags (ADR 0016);
- the MLX image generation engine (`x/imagegen`), which upstream removed in #16615;
- ADR 0017's `ClaimOSThread` and ADR 0018's per-thread imagegen streams.

Per ADR 0006 it is never merged into `main`; fixes flow one way, cherry-picked.

**This supersedes ADR 0019 and its branch `release/imagegen-mlx`.** That lineage was cut from the
same commit and preserves the same tree, but its ADR framed it as imagegen-only, which understates
what it protects and would let a reader conclude vision was unprotected. One lineage, correctly
described, is better than two identical ones. `release/imagegen-mlx` should be deleted once this
branch is confirmed.

## Alternatives considered

- **Keep both lineages.** They are byte-identical apart from their ADR, so the second buys nothing
  and invites drift between two branches nobody has decided how to maintain differently.
- **Rely on `release/imagegen-mlx` alone.** It does preserve everything, but only by accident of
  having been cut from `main`. A future reader searching for the vision decision finds an ADR about
  image generation and reasonably concludes vision was not considered.
- **Do not branch; port gemma4 before merging upstream.** The right end state, and it is the
  planned next step — but it is model-layer work whose only real validation is hardware-gated, so
  the safety net should exist before the merge lands, not after.

## Consequences

- Positive: nothing the fork built is lost when `main` follows upstream, and the record says which
  branch holds what and why.
- Negative: this lineage ages against `main` from the moment upstream lands, and the gap is
  precisely the media rework it cannot absorb without the gemma4 port.
- Negative: as with ADR 0019, `main` carries no record of what it gave up. Accepted; the pointer is
  this branch.
- Follow-up: port gemma4 onto `base.MediaModel` on its own branch, with the vision suites actually
  run on an Apple Silicon host. Until that lands, the upstream merge must not be pushed.

## Conformance

- `release/mlx-vision` is at or after `a8a25886` and contains `x/models/gemma4/vision.go`
  declaring `base.VisionModel`, `visionPrefillMaskBudget` and `prefillChunkLen` in
  `x/mlxrunner/pipeline.go`, both ADR 0014 conformance tests in
  `x/mlxrunner/client_format_test.go`, and the `x/imagegen` tree.
- `go test -p 1 -count=1 ./x/...` is green on this lineage, `x/imagegen` included.
