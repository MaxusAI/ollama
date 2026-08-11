# Plan: port gemma4 onto upstream's `base.MediaModel`

**Date:** 2026-08-11
**Why now:** it is the one blocker between the fork and `upstream/main`. See ADR 0020
(`release/mlx-vision`) for the decision this unblocks.
**Branch:** cut from the upstream merge, not from `main` — the target API only exists there.

---

## What's already true (verified 2026-08-11, not assumed)

**The target interface** — `x/mlxrunner/model/base/media.go` on the merged tree:

```go
type MediaModel interface {
    PrepareMedia(segments []Segment) (*PreparedRequest, error)   // CPU, request goroutine, deterministic
    EncodeMedia(item *PreparedItem, data *mlx.Array) *mlx.Array  // MLX thread, lazy, must not evaluate
}
```

`Segment{Tokens, Kind, Data}`. `PreparedItem{Range [2]int, Source int, MediaData []float32, Dims []int,
Opaque any, Causal bool}`. `PreparedRequest{Tokens, Items, Layout any}`.

**The runner gates all media on it.** `pipeline.go` returns `this model does not support %s input`
for any model that is not a `base.MediaModel`.

**gemma4 does not implement it.** It implements the fork's `base.VisionModel`:
`SupportsVision` (vision.go:459), `VisionTokens` (464), `NewVisionInput` (474), `EncodeVision` (538),
`MergedEmbeddings` (554). After the merge every image request against gemma4 400s, while
`x/create/client/create.go:337` still advertises the vision capability.

**The reference implementation is glimmer** — `x/models/glimmer/media.go`: `PrepareMedia` splices
`image_start` + patch-token run + `image_end` and records one `PreparedItem`; `EncodeMedia` runs the
tower; `scatterMedia` (media.go:93, called from glimmer.go:551) overwrites the placeholder rows in
the hidden state.

**The load-bearing difference, and the main risk in this plan:** glimmer sets `Causal: true`
(media.go:71) and so does qwen3_5 (vision.go:265). **No upstream model sets `Causal: false`.**
gemma4's image spans are bidirectional, so gemma4 will be the *first* consumer of upstream's
non-causal media path — a path upstream itself does not exercise. Budget for finding bugs in it.

**Today's bidirectional mask is offset-dependent.** gemma4 gates the bidi path on
`b.SeqOffsets[0] == 0` (gemma4.go:1266) and `visionChunkMask` indexes the key axis as absolute
prompt positions. ADR 0014 kept that honest by forcing an image block onto a position-zero chunk
(`prefillChunkLen`) and by refusing prompts whose opening chunk would need a >1 GiB dense mask
(`checkVisionPrefillBudget`). **Upstream's replacement does not preserve that.**
`media.extendChunk` only prevents a chunk *ending* inside an atomic expansion; it explicitly
accepts a resume point inside one. So the mask must become offset-aware, or bidirectional attention
is silently lost — ADR 0014's own words: this "degrades vision quality *silently* rather than
failing a test."

**The suites can actually run here.** `OLLAMA_VISION_E2E=1` gates both
`x/mlxrunner/vision_e2e_test.go` (shape + spatial, default `gemma4:12b-nvfp4`) and
`vision_golden_test.go` (`TestVisionGoldenParity` against `testdata/vision_goldens_{12b,26b,31b}.json`).
All three model sizes are present in `~/.ollama/models-mlx` (`12b-nvfp4`, `26b-nvfp4`, `31b-nvfp4`).
The preflight harness is `docs/maxusai/vision-suite/preflight/`.

**Also broken by the merge, in scope here because it is the same subsystem:** ADR 0014's two named
conformance tests (`TestPrefillChunkLen`, `TestCheckVisionPrefillBudget`) no longer compile;
`batch.InputsEmbeds` / `BidiSpans` have no writer; runner-level audio rejection is gone;
`prefix_cache`'s restore floor survives with zero tests; and the new `x/models/glimmer` package
drives MLX in tests with no `ClaimOSThread` (ADR 0017).

---

## Definition of Ready

Holds now: the interface is read, the reference implementation is identified, the failure mode is
understood, and the verification hardware and models are on hand. Two open questions below do not
block starting — they block finishing phase 4.

## Phases

Each phase ends with a **runnable** gate. Do not advance on "it compiles".

### Phase 1 — gemma4 becomes a `base.MediaModel`, text path unchanged

Implement `PrepareMedia` over gemma4's existing preprocessing (reuse `NewVisionInput`'s decode,
resize and compositing — ADR 0015's alpha-over-white must survive verbatim). Splice the same
placeholder token run gemma4 expects today. Set `Causal: false`. Record soft-token count and
geometry in `Opaque`. Leave `EncodeMedia` returning a zero tensor of the right shape.

> **Gate:** `go build ./x/... && go test -p 1 -count=1 ./x/mlxrunner/... ./x/models/gemma4/`
> green, and a text-only request through the runner is byte-identical to before the port.
> Image requests must now reach `EncodeMedia` instead of 400ing — assert that explicitly.

### Phase 2 — real features: `EncodeMedia` + scatter

Port `EncodeVision` into `EncodeMedia` (lazy — must not evaluate; the consuming forward pulls it).
Replace `MergedEmbeddings` / `b.InputsEmbeds` with a gemma4 `scatterMedia` modelled on
glimmer's, driven by `batch.Batch.Media` ranges.

> **Gate:** `OLLAMA_VISION_E2E=1 go test -run TestVisionGoldenParity -count=1 ./x/mlxrunner/`
> passes on **12b**. This is the real gate for this phase: the goldens compare against the vendored
> mlx-vlm reference, so wrong scatter, wrong axis or a lost scale fails it. Do not proceed on a
> passing build.

### Phase 3 — offset-aware bidirectional mask (the risky one)

Replace `BidiSpans` and the `SeqOffsets[0] == 0` gate with `nn.AttentionMask.Relax(seq, qLo, qHi,
kLo, kHi)` rectangles derived from each item's `Range`, expressed **relative to the chunk's own
offset** so a block riding a mid-prompt chunk still attends bidirectionally. Delete
`visionChunkMask`'s absolute-position assumption.

> **Gate:** both, on hardware:
> 1. `OLLAMA_VISION_E2E=1 go test -run 'TestVisionGoldenParity|TestVisionEndToEnd' -count=1 ./x/mlxrunner/`
>    on 12b **and** 26b — the `spatial` subtest is the one that catches a broken bidi mask, since a
>    causal-only image span still names shapes but loses left/right placement.
> 2. A new unit test that drives the same image at a **non-zero** chunk offset and asserts the mask
>    matches the offset-zero case. This is the regression the old `SeqOffsets[0] == 0` gate made
>    impossible to hit, and it must exist before this phase is called done.

### Phase 4 — decide ADR 0014's admission ceiling

Either reinstate `checkVisionPrefillBudget` against upstream's `mediaItem` ranges, or record that
`extendChunk`'s bounded growth plus an offset-aware mask subsumes it. **Do not drop it silently —
ADR 0014 called it a DoS fix.** See Open Questions.

> **Gate:** a test proving an adversarial prompt (image at the far end of a max-length context)
> either 400s cleanly or allocates a bounded mask. Measure the allocation; do not assert by eye.

### Phase 5 — remove the dead fork surface

Delete `base.VisionModel`, `VisionInput`, `NewVisionInput`/`EncodeVision`/`MergedEmbeddings`/
`SupportsVision`, `batch.InputsEmbeds`/`BidiSpans`, and `Request.VisionInputs`/`VisionSpans`/
`CacheSalts`. Port `vision_e2e_test.go`'s assertions onto `MediaItems`/`Layout`. Restore audio
rejection as a `seg.Kind != "image"` check in gemma4's `PrepareMedia`. Add a `skipIfNoMLX` helper
to `x/models/glimmer` (ADR 0017). Re-add coverage for `prefix_cache`'s restore floor.

> **Gate:** `go test -p 1 -count=1 ./x/...` green with **zero** skipped vision tests when
> `OLLAMA_VISION_E2E=1`, plus `grep -rn "VisionInputs\|InputsEmbeds\|BidiSpans" x/` returning
> nothing outside git history.

### Phase 6 — land

Run the preflight harness (`docs/maxusai/vision-suite/preflight/preflight.py`) against a build on
the Apple Silicon host and diff against `expectations.toml`. Write the ADR recording what the port
changed and which ADR 0014 guarantees survive in what form. Then, and only then, the upstream merge
becomes landable.

> **Gate:** preflight PASS with no `NEEDS_BASELINE` or `CONTENTION`, and a benchmark comparison per
> ADR 0012 showing vision tok/s within noise of `release/mlx-vision`.

---

## Open questions — answer before phase 4, do not guess

1. **Can the per-request image-token budget survive?** ADR 0003/0007/0008 let `image_max_tokens`
   reach preprocessing per request. Upstream's `PrepareMedia(segments)` takes **no options**, so the
   budget can only come from load-time config, which is what ADR 0016's reload-on-resolved-flags
   already does. Is load-time-only acceptable, or does `PrepareMedia` need a model-scoped budget set
   at load? This also decides whether the prefix-cache budget separation still holds: upstream keys
   on `Dims`, which for gemma4 changes with the budget — but only because gemma4 resizes. A model
   that pooled to a token budget without changing pixel dims would alias.
2. **Ceiling or no ceiling** (phase 4) — depends on whether the phase 3 mask is genuinely
   offset-safe at every resume point, which phase 3's gate 2 answers.

## Out of scope

- Porting `x/imagegen` — it is deleted on `main` per ADR 0019/0020 and lives on `release/mlx-vision`.
- The flaky `TestSchedRequestsMultipleLoadedModels` (macOS and Windows) — unrelated, its own fix.
- `go vet`'s `x/imagegen/mlx/compile.go:84` `unsafe.Pointer` finding — moot on `main` after the merge.
- Upstreaming the `x/mlxrunner` thread-affinity fix — separate, drafted already.

## Rollback

Every phase is additive until phase 5. If the port stalls, `release/mlx-vision` holds the working
vision line and the upstream merge simply does not land — the fork stays where it is today, which
is a working state.
