# Plan: port gemma4 onto upstream's `base.MediaModel`

**Date:** 2026-08-11
**Why now:** it is the one blocker between the fork and `upstream/main`. See ADR 0020
(`release/mlx-vision`) for the decision this unblocks.
**Branch:** cut from the upstream merge, not from `main` — the target API only exists there.

---

## What's already true (re-verified against `upstream/main` = `400164d4`, 2026-08-11)

**Re-grounding result:** every claim below still holds at the new tip. Upstream moved two commits
(`bb7bba88 mlx: implement Nemotron 3 Nano Omni`, `400164d4 parsers: ...`). Nemotron Omni landed as
`x/models/nemotron_h` and does **not** implement `base.MediaModel` and carries no audio or vision
path, so it adds no second media kind and does not disturb the `imageMinTokens`/`imageMaxTokens`
naming. `base.MediaModel` is byte-identical, no model sets `Causal: false`, glimmer still pins
`maxImageTokens = 4096`, and the budget fields are still fork-only. What *did* go stale is the merge
base: those commits touch `pipeline.go`, `prefix_cache.go`, `prefix_cache_test.go`, `cache_trie.go`,
`mtp_test.go` and `nn/recurrent*.go`, so the WIP merge needs bringing forward before phase 1 starts.

**Upstream has independently hit ADR 0017's bug and shipped a partial fix.** New in `v0.32.7`:
`x/internal/mlxtest` with `Setup(t)`, whose comment reads "The thread pin is load-bearing, not
defensive: MLX's default stream cache is thread-local, and anything that migrates the goroutine
mid-test ... otherwise panics with *There is no Stream(gpu, 0) in current thread*." Same bug, same
message, found independently.

It is **weaker than ours in a way that matters**, and this is a resolution decision for the merge,
not a curiosity:

- `mlxtest.Setup` pairs `runtime.LockOSThread()` with `t.Cleanup(runtime.UnlockOSThread)`. Releasing
  the pin hands a thread back to the runtime's pool with MLX's thread-local stream state still on it
  — the precise hazard ADR 0017 documents and the direct cause of the `x/imagegen` SIGSEGV in ADR
  0018, where every test passed alone and any two in sequence crashed.
- Upstream still carries **both** root causes, verified at `400164d4`: `mlxCall` still takes
  `LockOSThread`/`defer UnlockOSThread` around a single call, and `DefaultStream()` still caches the
  thread-local stream in a process-global. There is no permanent claim or ownership flag anywhere.
  So `Setup` suppresses only intra-test migration; across tests the global cache still hands one
  thread's stream to another, and passes only when the scheduler reuses a thread.

**Decided convergence:** adopt upstream's `x/internal/mlxtest` as the shared seam — every fork test
helper calls `mlxtest.Setup(t)` instead of its own `skipIfNoMLX` body — but implement `Setup` with
`mlx.ClaimOSThread()` and **no** unlock. That keeps upstream's API so future merges stay clean,
collapses the fork's ten hand-rolled helpers into one place, and keeps the stronger guarantee.
`ClaimOSThread` is idempotent, so repeated calls cannot run the runtime's lock counter away.

## Originally verified 2026-08-11 (unchanged unless noted above)

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

**gemma4 is not the only MLX vision model after the merge.** On fork `main` today it is — no other
`x/models/*` package has a `vision.go` or `media.go`. But the merge brings two more, both already
`base.MediaModel`: `x/models/qwen3_5` (upstream's `1e85fe8e qwen3_5: image input support`, registering
`Qwen3_5ForConditionalGeneration`, with `PrepareMedia` at vision.go:207) and `x/models/glimmer`.
So the post-merge world has three MLX media models, of which **only gemma4 is unported** — and only
gemma4 would know about the fork's image-token budget. That asymmetry is a live design problem, not a
deferred one; see Open Questions.

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

**The image-token budget is fork-only and must survive.** `ImageMinTokens` / `ImageMaxTokens`
(`api/types.go:632-633`, defaults 70 / 1120 per ADR 0008) exist **only in this fork** — upstream has
neither. They are declared inside `Runner` ("options which must be set when the model is loaded"),
but the two paths honour them differently today: `llm/llama_server.go:1067,1102` resolves them at
load into CLI args, with per-architecture defaults for nemotron and qwen-VL, whereas the MLX path
reads them **per request** by passing `api.Options` straight into `vm.NewVisionInput(m.Data, opts)`.
Upstream's `PrepareMedia(segments)` takes no options, so that per-request path has no seam to arrive
through. They survived the merge in `api/types.go`; what did not survive is the way they reach
preprocessing.

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

### Phase 0 — bring the merge forward and converge the test seam

Re-merge `upstream/main` at `400164d4` onto the WIP merge branch (an incremental merge of two
commits; last attempt produced seven small conflicts: `server/images.go`, `images_test.go`,
`model_list_cache.go`, `x/mlxrunner/mtp_test.go`, `prefix_cache_test.go`, `x/models/nn/nn_test.go`,
`recurrent_test.go`). Apply the convergence above: rewrite `x/internal/mlxtest.Setup` to use
`mlx.ClaimOSThread()` with no unlock, and resolve every conflicting `skipIfNoMLX` toward
`mlxtest.Setup(t)`.

> **Gate:** `go build ./x/... ./server/... ./llm/...` clean, and `go test -p 1 -count=1 ./x/...`
> no worse than the WIP baseline. Note the WIP's known test-build break
> (`client_format_test.go` calling the deleted `prefillChunkLen` /
> `checkVisionPrefillBudget`) is phase 5's to resolve, not phase 0's — do not delete those tests
> here just to get green.

### Phase 1 — gemma4 becomes a `base.MediaModel`, with the budget threaded through

Implement `PrepareMedia` over gemma4's existing preprocessing (reuse `NewVisionInput`'s decode,
resize and compositing — ADR 0015's alpha-over-white must survive verbatim). Splice the same
placeholder token run gemma4 expects today. Set `Causal: false`. Record soft-token count and
geometry in `Opaque`. Leave `EncodeMedia` returning a zero tensor of the right shape.

**`image_min_tokens` and `image_max_tokens` are passed through per request** — decided, see below.
Upstream's `PrepareMedia(segments)` has nowhere to carry them, so add a **fork-local optional
interface** rather than changing upstream's:

```go
// x/mlxrunner/model/base — fork-local; upstream models are unaffected.
//
// The budget is named per media kind on purpose: a Segment may be text, an
// image or audio, and only images carry a token-rung budget today. Bare
// minTokens/maxTokens would claim a scope this does not have. The names also
// match the API fields they carry (api.Options.ImageMinTokens/ImageMaxTokens).
type MediaBudgetModel interface {
    PrepareMediaWithBudget(segments []Segment, imageMinTokens, imageMaxTokens int) (*PreparedRequest, error)
}
```

If a second media kind ever needs its own budget, collapse the pair into a
`MediaBudget` struct rather than growing the parameter list — but not before, since
there is no second kind to design against.

`pipeline.go` prefers it when the model implements it and falls back to `PrepareMedia` otherwise.
Keeping upstream's interface untouched means future merges conflict on one added file rather than
on every media model. Resolve the budget the way `llm/llama_server.go:1067` already does —
per-architecture defaults, request value wins — so the MLX and llama-server paths agree.

Two consequences that are requirements, not notes:

- **`PreparedItem.Dims` must reflect the resolved budget.** It feeds upstream's prefix-cache fold,
  so this is what keeps two budgets over the same image bytes from sharing a cache prefix. gemma4
  resizes per budget, so Dims does change — assert it rather than assume it.
- **Upstream's determinism contract widens.** It says `PrepareMedia` must be deterministic "for
  given segments". With a per-request budget it is deterministic for *segments + budget*, and the
  budget must therefore be part of cache identity. Anything that restores KV captured at a
  different budget is a correctness bug, not a cache-efficiency one.

> **Gate:** `go build ./x/... && go test -p 1 -count=1 ./x/mlxrunner/... ./x/models/gemma4/`
> green; a text-only request is byte-identical to before the port; image requests reach
> `EncodeMedia` instead of 400ing; and a test proves that the **same image bytes at two different
> budgets** produce different `Dims` and never share a prefix-cache prefix.

### Phase 1b — port `qwen3_5` and `glimmer` onto the budget seam

Add `PrepareMediaWithBudget` to both upstream models, delegating to their existing preprocessing
with the resolved ceiling substituted for their hard-coded constant. Each keeps its own default:
a request value equal to the shared default counts as unset, per `llm/llama_server.go:1102-1113`.
Keep the diff minimal and mechanical — these are upstream files and every future merge will conflict
here, so the smaller and more obviously-shaped the change, the cheaper each resolution.

Then `pipeline.go` can treat `MediaBudgetModel` as the expected interface rather than an optional
one, since all three MLX media models implement it.

> **Gate:** `go test -p 1 -count=1 ./x/models/glimmer/ ./x/models/qwen3_5/` green, plus a test per
> model proving that (a) an unset/default budget reproduces the model's own ceiling byte-for-byte —
> glimmer must still resolve 4096 — and (b) an explicitly different budget changes the resulting
> `Dims`. Without (a) this phase silently degrades OCR, which is exactly what glimmer's comment
> warns about.

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

## Decided

**The image-token budget is passed through per request.** `image_min_tokens` and
`image_max_tokens` must reach preprocessing on every request, not be frozen at load. Load-time-only
was the cheaper option and is rejected: it would silently demote a documented, fork-only API field
to a restart-scoped one, and ADR 0008's whole finding is that the right rung differs by model *and*
by task — the 12B prefers 560 for bbox work while 26B/31B want 1120, which is a per-request choice.

Implemented as the fork-local `MediaBudgetModel` seam in phase 1. Cost, stated plainly: one more
fork-local interface to carry across every upstream merge, and a determinism contract that is now
segments + budget rather than segments alone. It is a clean addition rather than a modification, so
it is upstreamable if upstream ever wants per-request budgets.

**Every MLX media model implements the budget seam — `qwen3_5` and `glimmer` are ported too.**
Decided against the two cheaper answers. Accept-and-warn is a silent drop with extra steps, which
ADR 0009 rejects outright. Rejecting with a 400 is honest but unusable in practice: `DefaultOptions`
always populates `ImageMinTokens`/`ImageMaxTokens`, so *every* request carries a budget and a gate
would 400 all qwen and glimmer image traffic. Porting all three is the only answer that keeps the
field meaningful everywhere.

`MediaBudgetModel` therefore becomes the expected interface for MLX media models rather than an
optional extra, and `pipeline.go` can gate on it. Two consequences:

- **A model's own ceiling stays its default.** glimmer pins `maxImageTokens = 4096` with a comment
  that lowering it discards detail and hurts OCR; that is a considered choice, not an oversight, and
  handing it gemma4's 1120 default would silently degrade it. Reuse the resolution convention
  `llm/llama_server.go:1102-1113` already uses for nemotron and qwen-VL: a value equal to the
  *shared* default counts as unset and the model's own default applies; only a genuinely different
  value overrides. That keeps the MLX and GGUF paths consistent for the same model family.
- **Cost: this is fork divergence inside upstream's own model files.** `x/models/qwen3_5` and
  `x/models/glimmer` are upstream code, so every future upstream merge will conflict there. Keep the
  change small and mechanical — one extra method delegating to the existing preprocessing — so the
  conflicts stay trivial to resolve.

## Open questions — answer before phase 4, do not guess

1. **Ceiling or no ceiling** (phase 4) — whether ADR 0014's 1 GiB admission ceiling must be
   reinstated depends on whether the phase 3 mask is genuinely offset-safe at every resume point,
   which phase 3's gate 2 answers. Do not resolve this before that gate runs.
*(The former question 2 — what happens when a budget reaches a model that cannot honour it — is
answered above: all three models are ported.)*

## Out of scope

- Porting `x/imagegen` — it is deleted on `main` per ADR 0019/0020 and lives on `release/mlx-vision`.
- The flaky `TestSchedRequestsMultipleLoadedModels` (macOS and Windows) — unrelated, its own fix.
- `go vet`'s `x/imagegen/mlx/compile.go:84` `unsafe.Pointer` finding — moot on `main` after the merge.
- Upstreaming the `x/mlxrunner` thread-affinity fix — separate, drafted already.

## Rollback

Every phase is additive until phase 5. If the port stalls, `release/mlx-vision` holds the working
vision line and the upstream merge simply does not land — the fork stays where it is today, which
is a working state.
