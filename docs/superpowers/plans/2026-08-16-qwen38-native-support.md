# Plan: native Qwen3.8 support

- **Status:** proposed, not started. Grounded against `main` at `677915ad` and
  `upstream/main` at `d67ad834` on 2026-08-16.
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

### Phase 0 — resolve the gating unknown (blocking)

Everything below rests on three unverified assumptions about a real checkpoint.
Pull one and check. This is inspection, not research — real tags are published.

- Does it declare a `Qwen3_5*` / `Qwen3Next*` architecture? If it declares
  `Qwen3_8*`, both `isQwen35Family` and `base.Register` miss; only the latter
  fails loudly.
- Do the template markers (`resolved_reasoning_effort`, `preserve_thinking`) live
  in `chat_template.jinja`? Upstream's detection reads only that. A checkpoint
  shipping a legacy `chat_template` inside `tokenizer_config.json` is detected as
  **qwen3.5 — wrong renderer, no error, model imports and serves.**
- If it has a `vision_config`, does it omit `deepstack_visual_indexes`? Our tower
  hard-rejects that (`vision.go:27-29`).
- For the GGUF flavour: what arch string does it declare? `qwen35`/`qwen35moe`
  means CUDA works immediately; anything else needs a llama.cpp-side change, and
  spec B4 warns that adding to `visionServerArgs` alone is half a change.

**Gate:** the four answers written down. Do not start Phase 1 without them.

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

### Phase 3 — settle measurement provenance before any benchmark

`ThinkValue.String()` returns `"medium"` for boolean `true`, so `think:true`
renders at **medium** effort while *omitting* `think` yields the publisher's
**xhigh** default — the explicit request is quieter than the implicit one. `xhigh`
is unreachable through our API (`IsValid()` accepts only high/medium/low/max), and
our preflight sends `think=True` (`checks.py:417`).

So a qwen3.8 number recorded as "model default" is in fact medium effort.

**Do not** add a `CARD_THINKING` entry by analogy to qwen3.6 — `family()` falling
through to packaged defaults is the *correct*, ADR-0023-admissible behaviour here,
because qwen3.8 ships its own effort default.

**Gate:** the effort label is decided and written into whatever record becomes a
baseline. This is a decision, not code.

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

1. Phase 2 fix shape: make truncation treat a renderer error as "unusable window,
   keep going" (general, protects every validating renderer, but touches the
   image-token accounting function), or downgrade qwen3.8's rejection to a warning?
2. Does the ADR 0004 pass-two continuation need a structural guarantee that the
   re-rendered window keeps a user turn? Pass two re-runs truncation on a strictly
   longer prompt, so a request that fit on pass one can fail mid-stream.
3. Phase 3: how is reasoning effort labelled in a baseline, given our practical
   default is medium and the publisher's is xhigh?
4. Phase 4: a `think_format` block for `apple-silicon-mlx`, or cheaper coverage as
   a Go-level format+think regression test through the qwen3.8 renderer?
5. Harden `qwen35RendererName` to read `chat_template.jinja` independently, as the
   laguna and nemotron paths already do? It is fork divergence on a line upstream
   keeps editing — the drift that produced our existing `qwen35.go` divergence.

## Caveat on a related record

ADR 0009 and ADR 0013 describe `x/structured` as a port of llama.cpp **b10091**.
That is accurate as provenance — b10091 was the pin when the port landed — but the
payload pin is now **b10353**, and the `MAX_REPETITION_THRESHOLD = 2000` check in
ADR 0013 was verified against a stale `build/_deps` checkout still at b10091. ADR
0013 already says the threshold tracks the pin; the value needs re-confirming on
the next payload build.
