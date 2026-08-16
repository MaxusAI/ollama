# Plan: native Qwen3.8 support

- **Status:** Phases 0–3 complete (2026-08-16); Phases 4–5 not started.
  Phase 1 shipped in #111, Phase 2 in #112, open question 2 in #113,
  Phase 3 as ADR 0026. Grounded against `main` at `677915ad` and `upstream/main` at `d67ad834`.
- **Scope:** land Qwen3.8 on the fork across the engines we ship, and know it works.
- **Not an ADR.** Decisions that get made while executing this belong in ADRs;
  this file is disposable once the work ships.

## What is already true

Upstream added Qwen3.8 on 2026-08-14 in three commits we do not have. Nothing
regressed on our side — `main` is 18 commits behind `upstream/main` and 291 ahead.

| commit | what it does |
| --- | --- |
| `55127975` | qwen3.8: add renderer and MLX import support (#17745) |
| `0f25c31b` | qwen3.8: support developer instructions (#17749) |
| `87abaa01` | renderers/qwen: tolerate non-leading system messages (#17757) |

The load-bearing fact: **Qwen3.8 keeps the Qwen3.5 architecture and parser.**
Upstream's entire model-layer change is a 4-line `Squeeze`→`Reshape` fix in
`sanitizeConvWeight`. This is not a port; it is a renderer, a detection rule, and
an import branch.

Consequences we verified rather than assumed:

- `x/models/qwen3_5` already runs this architecture, including MTP self-draft,
  packed GDN projections, packed gate_up experts and per-expert quantization.
  Nothing is version-gated against a 3.8 config.
- Qwen3.8 keeps the **parser name** `qwen3.5`; only the *renderer* name changes.
  It therefore inherits the fork's `Qwen35Parser.ThinkingCloseMarker()` and lands
  in ADR 0004's **marker flow**, not the transition flow. Spec R4 holds because
  it is literally the same parser object.
- Today, pulling a Qwen3.8 model and chatting fails with
  `unknown renderer "qwen3.8"` — `rendererForName` has no case. That single gap
  is what the cherry-picks close.
- `x/structured` (ADR 0009) needs no changes. Note `format` + MTP means no
  speculative speedup: a constrained request decodes serially and the speculation
  session never proposes.

## How each engine gets it

The renderer and parser live at the routes/model layer and are **engine-agnostic**:
a GGUF model takes its renderer from the manifest's `Config.Renderer`, so the same
cherry-picks serve llama-server and MLX alike. Only the `x/create` import half is
MLX-specific.

| | **CUDA** | **ROCm (gfx1151)** | **Apple Silicon MLX** | **Apple Silicon Metal** |
| --- | --- | --- | --- | --- |
| Engine | llama-server subprocess (GGUF) | llama-server subprocess (GGUF) | `x/mlxrunner`, in-process (safetensors) | llama-server subprocess (GGUF) |
| Model impl | llama.cpp `qwen35`/`qwen35moe` — present in the payload | b9888 payload — unverified | `x/models/qwen3_5` — present | llama.cpp `qwen35`/`qwen35moe` |
| The 3 picks give | renderer + parser | — (not backported) | renderer + parser + `x/create` import | renderer + parser |
| How a model arrives | `ollama pull` | `ollama pull` | `ollama pull` **or** `x/create` import | `ollama pull` |
| Preflight arches today | `nemotron_h_omni`, `gemma4` | `nemotron_h_omni`, `gemma4` | `gemma4`, `gemma4_unified`, `qwen35moe` | `gemma4`, `qwen35moe` |
| Status after Phase 1 | works, unmeasured | **blocked — see below** | works, unmeasured | works, unmeasured |

**`ollama pull` works for MLX.** `Model.IsMLX()` is `Config.ModelFormat ==
"safetensors"` (`server/images.go:84`), a manifest field, so a pulled model
self-declares as MLX and routes to `x/mlxrunner`. `x/create` is the alternative
path for local or unpublished checkpoints, not a prerequisite.

**ROCm is doubly blocked, and neither block is about Qwen3.8.** ADR 0006: `main`
is *not deployable* on gfx1151 while the AMD upgrade gate holds — b10091+ produced
degenerate vision output there and was rolled back 2026-07-31. The deployable AMD
line is `release/0.32.1-dynres` at b9888, whose `rocm-0-32-1-dynres` preflight
profile carries no `qwen35` family arch at all. ROCm support is gated on the AMD
gate lifting, which is outside this work.

## Landing strategy

Both options were measured in a scratch worktree, not argued:

| | result |
| --- | --- |
| **B — cherry-pick the three commits** | **zero conflicts**, builds and tests green; 24 files, +1882/−58 |
| A — merge `upstream/main` wholesale | no textual conflicts, but one **compile break** at `llm/llama_server.go:2184`, and drags in the llama.cpp and MLX payload bumps |

**Take B.** The decisive reason is not tidiness: a payload bump forces a
re-measurement (preflight `check_payload_pin` compares the running build sha
against a pinned value) and would confound every number produced during a model
bring-up. Keep the payload still.

Two conflicts reported during investigation are **not real** and should not be
budgeted for: the `x/models/qwen3_5/qwen3_5_test.go` collision is an artifact of
`git apply` versus cherry-pick's 3-way merge, and the "xhigh reasoning collides
with ADR 0023" claim does not survive contact with the code.

## Phases

### Phase 0 — resolve the gating unknown — **DONE 2026-08-16, all clear**

Answered by reading the published registry manifests and their config/JSON layers
directly (no weight download):
`curl https://registry.ollama.ai/v2/library/qwen3.8/manifests/{latest,27b-mxfp8}`
then fetching the `config` blob and the small `json` layers by digest.

| question | answer | verdict |
| --- | --- | --- |
| Architecture string | `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5` | **holds** — `isQwen35Family` and `base.Register` both match; no alias needed |
| Template markers | `resolved_reasoning_effort` **and** `preserve_thinking` both present | **holds** |
| …and reachable by detection? | they live in `tokenizer_config.json`'s `chat_template`; there is no separate `chat_template.jinja` layer | **holds** — `readChatTemplate` reads `tokenizer_config.json` *first*, then falls back to the `.jinja` file |
| `deepstack_visual_indexes` | present but **empty** (`[]`) | **holds** — our tower rejects only a *non-empty* value |
| GGUF arch | `model_family: "qwen35"`, `model_families: ["qwen35"]` | **holds** — llama.cpp already supports `qwen35`; CUDA needs no payload change |

Published config blobs, verbatim:

```json
// 27b-mxfp8 (MLX)
{"model_format":"safetensors","file_type":"mxfp8","renderer":"qwen3.8",
 "parser":"qwen3.5","requires":"0.32.12",
 "capabilities":["completion","vision","tools","thinking"]}

// latest (GGUF)
{"model_format":"gguf","model_family":"qwen35","model_families":["qwen35"],
 "model_type":"27.3B","file_type":"Q4_K_M","renderer":"qwen3.8",
 "parser":"qwen3.5","requires":"0.32.12"}
```

What this changes:

- **`parser: "qwen3.5"` is confirmed from the shipped artifact**, not inferred. The
  marker-flow inheritance and the R4 argument now rest on published metadata.
- **`model_format: "safetensors"` confirms `ollama pull` routes to the MLX runner**
  — `IsMLX()` reads exactly this field.
- **The renderer comes from the manifest for pulled models.** Template-marker
  detection only runs on `x/create` import of a local directory, which narrows the
  blast radius of any detection bug to that path.
- **The "silent wrong renderer" risk is refuted for this checkpoint.** The concern
  assumed `readChatTemplate` consults only `chat_template.jinja`; it consults
  `tokenizer_config.json` first, which is where this model's template lives. The
  residual is hypothetical — a checkpoint shipping a *stale* template in
  `tokenizer_config.json` alongside a fresh `.jinja`. This one does not.
- **`requires: "0.32.12"` is not a gate.** `Config.Requires` is only surfaced by
  `ollama show` (`cmd/cmd.go:1344`) and propagated on create; nothing enforces it
  on pull. Our `version.Version` dev default of `0.0.0` will not block anything.
- Vision preprocessing is `Qwen3VLProcessor` / `Qwen2VLImageProcessorFast`,
  patch 16, merge 2, mean/std 0.5 — relevant to Phase 5, not to landing.

**Phase 1 is unblocked.** Every assumption the plan rested on held.

### Phase 1 — land the cherry-picks

Three picks on a branch off `main`.

**Gate:** CI's existing `test:` job already runs the new `qwen38_test.go`,
renderer, parser and `x/create` tests — no new harness needed. Locally,
`go test ./model/renderers/... ./model/parsers/... ./x/create/... ./server/...`.
Note `go build ./...` is known-broken here (`app/dist` embed); build the touched
packages instead.

### Phase 2 — fix the fork-specific truncation defect

Qwen3.8's `validateMessages` runs **inside `chatPrompt`'s truncation loop**. With
ADR 0004's pass-two continuation, the error is emitted mid-stream *after* pass-one
content already reached the client (`routes.go:3337-3341`, `:937-941`), and ADR
0010's metrics reconstruction never completes — a truncated stream ending in an
error. Upstream has no equivalent flow, so **upstream's tests cannot see this.**

Engine-agnostic: this is a routes-layer flow, so it affects llama-server and MLX
equally.

**Gate:** a regression test through the qwen3.8 renderer exercising format+think,
in the CI-gated job. It must fail before the fix.

### Phase 3 — settle measurement provenance — **DONE 2026-08-16**

Recorded as [ADR 0026](../../maxusai/adr/0026-qwen38-baselines-record-the-effort-directive.md)
and pinned by `TestQwen38ReasoningEffortMapping`
(`model/renderers/qwen38_effort_test.go`).

**This phase's original premise was wrong, in the direction that would have
invalidated a benchmark.** It claimed `think:true` renders at a quieter
"medium" while omitting `think` yields the publisher's xhigh — the explicit
request weaker than the implicit one. Measured by rendering rather than
reading:

| `think` | directive emitted |
| --- | --- |
| `nil` | **xhigh** |
| `true` | **none** |
| `"medium"` | **none** |
| `"low"` | low |
| `"high"` / `"max"` | **xhigh** |

Two corrections:

- `"medium"` is **not a middle setting** — the renderer emits *no directive*
  for it, so `think:"medium"` and `think:true` are the same prompt.
- The `nil` row is **unreachable over HTTP**. `server/routes.go` coerces a nil
  think to `true` for thinking-capable models (`:462`, `:2927`), and Qwen3.8
  declares `thinking`. Omitting `think` and sending `think:true` are therefore
  byte-identical; there is no explicit-is-quieter trap.

What survives: the API default emits **no effort directive**, so a run
labelled "model default" must say so rather than implying the publisher's
xhigh — which is reachable only via `think:"high"`/`"max"`. The preflight
harness sends `think=True`, so its runs *are* the no-directive default and
need no change; the earlier worry that preflight secretly measured "medium"
does not hold.

The `CARD_THINKING` guidance stands unchanged: do **not** add a `qwen3.8`
entry by analogy to qwen3.6.

### Phase 4 — preflight baselines (ADR 0011)

Two targets, not one: `apple-silicon-mlx` **and** `cuda-dynres-005`. Neither has a
`qwen35moe` row today, and ADR 0011 rule 4 makes an unmeasured (platform, arch)
combination a `NEEDS_BASELINE` failure. Measure; never copy a row across.

Related gap: the `think_format` probe **skips on apple-silicon-mlx** — the exact
profile where Qwen3.8 runs natively, and the flow Phase 2's defect lives in.
Adding a block to the `qwen35moe` arch also changes what that existing measured
row asserts for qwen3.6, so it is a change to a measured row, not a pure addition.

### Phase 5 — vision registration

Upstream registers qwen3.8 for **tools only** — it is absent from
`releaseVisionModels` and `releaseVisionTextModels` despite the manifest declaring
vision capability and shipping a projector. For a fork whose differentiator is
vision, taking upstream's registration verbatim leaves that untested.

Splits by engine: MLX goes through `x/models/qwen3_5` vision; GGUF goes through
`visionServerArgs` and the image-token B-invariants, where `qwen35`/`qwen35moe`
already carry the 1024 floor.

## Out of scope

- **Payload bumps** (llama.cpp b10353→b10434, MLX 8c28c385→adf21dea) and the other
  11 upstream commits. None help Qwen3.8; all cost re-measurement. If taken, note
  `test-llamacpp-update.yaml` is inert on this fork — the real coverage is
  `test.yaml`'s ungated `patches:` job.
- **Backporting to `release/0.32.1-dynres`.** That lineage has no `qwen35` family
  arch row in its rocm profile, so the family is unmeasured there. Worth recording
  as a deliberate "no" under ADR 0006 so it is not reopened as a merge.

## Open questions

1. ~~Phase 2 fix shape?~~ **Closed by #112**: neither option. "Keep going"
   would only defer the failure, because windows shrink as truncation
   advances — the fix keeps the smallest *renderable* window instead.
2. ~~Does the ADR 0004 pass-two continuation need a structural guarantee?~~
   **Closed by #113**: pass two now pins itself to pass one's truncation
   window, and an over-long pinned window becomes a clean `length` exit via
   the existing continuation-headroom check rather than a silent context loss.
3. ~~Phase 3: how is reasoning effort labelled in a baseline?~~ **Closed by
   ADR 0026.** The premise was wrong — the API default emits *no* directive,
   not "medium", and omitting `think` is identical to `think:true`. Baselines
   record the emitted directive.
4. Phase 4: a `think_format` block for `apple-silicon-mlx`, or cheaper coverage as
   a Go-level format+think regression test through the qwen3.8 renderer?
5. ~~Harden `qwen35RendererName` to read `chat_template.jinja` independently?~~
   **Closed by Phase 0.** `readChatTemplate` already reads `tokenizer_config.json`
   first and falls back to `chat_template.jinja`, and the shipped checkpoint keeps
   its template in the former. No fork divergence needed — which is the better
   outcome, since it would have sat on a line upstream keeps editing.

## Caveat on a related record

ADR 0009 and ADR 0013 describe `x/structured` as a port of llama.cpp **b10091**.
That is accurate as provenance — b10091 was the pin when the port landed — but the
payload pin is now **b10353**, and the `MAX_REPETITION_THRESHOLD = 2000` check in
ADR 0013 was verified against a stale `build/_deps` checkout still at b10091. ADR
0013 already says the threshold tracks the pin; the value needs re-confirming on
the next payload build.
