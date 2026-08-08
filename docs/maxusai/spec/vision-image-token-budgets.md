# SPEC: vision image-token budgets

MaxusAI-fork specification. Status: **implemented** on both maintained lineages
(fork `main`, `release/0.32.1-dynres`). Written 2026-08-02.

Normative contract for how many visual tokens an image costs and who decides it.
Measured costs per arch and size live in
`vision-token-budget-measurements.md` (main lineage); the
mechanism analysis is in
`vision-token-budgets-by-arch.md` (main lineage); the decisions
are ADR 0001 (`0001-nemotron-vision-dynamic-resolution.md`, main lineage) and
[ADR 0003](../adr/0003-vision-image-token-budget-policy.md).

## 1. Where the budget is decided

**B1 — Budgets are launch-time.** `visionServerArgs(modelArch, opts)` contributes
`--image-min-tokens` / `--image-max-tokens` to the llama-server process command line
when the runner starts. Budgets are therefore a property of the **loaded runner**, not
of a request.

**B2 — Budgets are endpoint-independent.** Because of B1, `/api/chat`,
`/api/generate` and every `/v1` endpoint MUST observe identical per-image costs for
the same model and image. Measurements taken through one endpoint are valid for all.
(Verified: nemotron3 + dynres, 1920×1080 image, `prompt_eval_count` 2061 on both
`/api/generate` and `/api/chat`.)

**B3 — Budget options are Runner options.** `ImageMinTokens` / `ImageMaxTokens`
changes reload the runner. Clients MUST expect a reload when they vary them per
request.

**B4 — Two sides must agree.** A flag only has effect if the loaded projector calls
`set_limit_image_tokens()` and consumes it. Adding an arch to `visionServerArgs`
is **half** the change; the other half belongs to llama.cpp (upstream or a
`llama/compat` patch). An arch whose projector ignores the flags MUST NOT be given
them, so the API does not advertise a knob that does nothing — with the deliberate
exception in B5.

**B5 — Forward-compatible flags are permitted where a lineage patch makes them
live.** `nemotron_h_omni` receives budget flags on every maintained lineage: both
currently carry `llama/compat/002-llama-cpp-nemotron-dynres.patch`, which consumes
them. Against a pristine `llama/` (upstream stock, or `main` between `5ad093b0` and
`2487dd56`) llama-server still parses them but the projector ignores them and the cost
is a structural 256/image. Whether the flags are live is therefore a property of the
payload, not of the arch — recorded in the table rather than silently tolerated.

## 2. Per-architecture contract

| `modelArch` | flags | effective budget | consumed by |
|---|---|---|---|
| `gemma4` | min/max from `api.Options`, defaults **70 / 1120** | 70 … 1,120, **budget-fill** (compat/004) | gemma4v projector, `set_limit_image_tokens(40, 280)` with the ceiling raised |
| `qwen2vl`, `qwen25vl`, `qwen3vl`, `qwen3vlmoe`, `qwen35`, `qwen35moe` | `--image-min-tokens 1024` | 1,024 … 4,096 | `PROJECTOR_TYPE_QWEN3VL`, `set_limit_image_tokens(8, 4096)` with the floor raised |
| `nemotron_h_omni` | min/max from `api.Options`, defaults 256 / 3328 | 256 … 3,328 with compat/002; exactly 256, flags inert, without it. A pinned budget no longer overshoots the ceiling (compat/005) | `PROJECTOR_TYPE_NEMOTRON_V2_VL` as patched (ADR 0001) |
| `mistral3`, `glmocr`, `llama4`, `deepseekocr`, all others | none | projector default / structural | — |

`qwen35` and `qwen35moe` are in the Qwen row because `llama/compat`'s
`handle_qwen35_like_clip()` sets `clip.projector_type = "qwen3vl_merger"`, so they
load as `PROJECTOR_TYPE_QWEN3VL` — the branch that emits the "requires at minimum
1024 image tokens" warning.

### 2.1 gemma4 budget-fill sizing (compat/004, on this lineage since `5f6e7fdc`)

**This changes what `min` means for gemma4.** Without 004 the minimum is a *floor* — a small
image is never enlarged, so it costs whatever its native grid costs and the budget goes
unused. With 004 the image is scaled **up or down** so its 48-aligned patch grid *fills* the
budget, snapped down to Gemma 4's supported ladder, and resized with `PAD_NONE` rather than
letterboxed.

- Supported ladder: **{70, 140, 280, 560, 1120}**. `DefaultImageMinTokens = 70` is the lowest
  rung, which makes `min` effectively a no-op at defaults; the ceiling does the work.
- Conformance (SPEC B7): every gemma4 grid must satisfy
  `cols·rows ≤ B < (cols+1)·(rows+1)` for a supported budget `B`. An off-ladder grid is a
  sizing defect — it measurably degrades `box_2d` vertical grounding, which is why 004 exists.
- 12B `box_2d` workloads should pin `image_max_tokens 560` per request (recorded exception,
  main ADR 0008).

### 2.2 Pinned-budget overshoot (compat/005, since `593fc3b1`)

When a pinned budget (`min ≈ max`) made the `min_pixels` ceil exceed `max_pixels`, sizing
floored just *above* the ceiling — nemotron pinned to 3328 delivered **3388**. 005 floors
just under `min` instead. The budget is a hard ceiling. Affects only the infeasible pinned
case; unpinned sizing is unchanged.

### 2.3 Measured on this lineage

Build `0.32.1-dynres-35d9e58e` (payload b9888 + 002/004/005), gfx1151/ROCm, deployed
2026-08-08. Image tokens are `prompt_eval_count` minus the model's **own** text-only
baseline minus 16 per image — the text figure is model-specific, so do not reuse another
model's:

> **Unverified against B8 — the nemotron rows below are suspect.** Two subtrahends are
> applied here and B8 questions both. The text-only baseline is the wrong one if this
> payload behaves like the main lineage's, and the flat **16** is `gemma4`'s marker count:
> `nemotron_h_omni` carries **2** markers per image, not 16, so subtracting 16 from a
> nemotron count reads ~14 low. Re-derive the prefix per §3 step 1 and the per-arch marker
> count on **this lineage's own build** before relying on the nemotron cells; the gemma4
> cells are internally consistent (both grids satisfy B7). Not corrected here because §4
> forbids importing another lineage's measurements.

| model | case | grid / tokens | conforms |
|---|---|---|---|
| gemma4:31b @1120 | 1920×1080 scene | 1100 = 20×55 | ✓ `1100 ≤ 1120 < 21×56` |
| gemma4:31b @1120 | 1568² document | 1089 = 33×33 | ✓ `1089 ≤ 1120 < 34×34` |
| nemotron3:33b | defaults | 2026 | — |
| nemotron3:33b | pinned 3328 | **3254** | ✓ ≤ ceiling (was 3388 pre-005) |

`qwen35`/`qwen35moe` are unaffected by 004 and 005 — measured byte-identical before and
after (`prompt_eval` 2615 scene / 2743 document) — since 004 is gemma4-only and 005 touches
shared `dyn_size` sizing that the Qwen path does not reach.

## 3. Verifying that a flag binds

Adding an arch to the switch MUST be accompanied by an empirical check, because B4
cannot be established by reading the Go side.

**B8 — Image-token measurements MUST cancel the text prefix, and MUST NOT take it from
a text-only request.** Every probe in the procedure below MUST send one identical
prompt; a baseline measured with a different prompt puts the text-length difference
into every result. Beyond that, the prefix can tokenise *differently once an image is
attached*, so the text-only count is the wrong subtrahend even when the prompt matches.
Measured on the **main** lineage (`0.32.5-dynres-4987dd49`, 2026-08-08),
`"Describe briefly."` costs 21 tokens text-only but 20 inside an image-bearing request
on `nemotron_h_omni`, while `gemma4:31b` costs 19 both ways. Per §4 that measurement
does not port to this lineage as a *value* — only the requirement ports. Derive the
offset per model and per payload; never assume it, and never carry it across lineages.

Procedure — the fingerprint method:

1. Calibrate the prefix from a two-image difference, which cancels it without assuming
   a grid: for one fixed prompt and two images A and B,
   `prefix = count(A) + count(B) − count(A, B)`, where `count` is `prompt_eval_count`
   at `num_predict: 1`. (Each of the three terms carries the prefix exactly once, twice
   summed minus once.) `vision-suite/measure.py` on the main lineage implements this.
2. Measure the same prompt with a **sub-budget** image (small enough that the
   proposed floor would bind, or large enough that a ceiling would). Image tokens are
   `prompt_eval_count − prefix`.
3. Repeat with the flag applied. The count MUST change in the predicted direction. If
   it does not, the projector is ignoring the flag and the arch MUST NOT be added
   (B4).
4. Re-measure at corpus-representative sizes to confirm the change is confined to the
   intended range.

Worked example (qwen3.6 `qwen35moe`, b9888 + 002, baseline 14). This one predates B8 but
survives it: its rows land on grid + 2 exactly (28×28 + 2 = 786, 49×49 + 2 = 2,403), which
is only possible if 14 was also the in-image prefix — so `qwen35moe`'s offset is 0.

| image | without floor | with `--image-min-tokens 1024` |
|---|---|---|
| 224×224 | 51 | 1026 |
| 448×448 | 198 | 1026 |
| 896×896 | 786 | 1026 |
| 1920×1080 | — | 2042 (floor does not bind) |
| 1568×1568 | — | 2403 (floor does not bind) |

## 4. Lineage rule

The maintained lineages pin different llama.cpp versions and patch them independently,
so the *consumers* of these flags can differ even though the Go-side switch is shared.
Consequences:

- Arch entries MAY be backported freely between lineages; **assertions about a
  projector's behaviour MAY NOT**. A change that adds an arch and also asserts what a
  *different* arch's projector does must be split, taking only the arch-specific half.
  (Concretely: `87cf1100` added the qwen floor *and* asserted `nemotron_h_omni` gets no
  flags. The first half was portable; the second described only `main`'s
  then-pristine `llama/` and was already false elsewhere.)
- Every lineage MUST keep `TestVisionServerArgs` expectations consistent with its own
  payload, and MUST re-check them whenever that payload changes. Expectations agreeing
  across lineages today is a coincidence of both carrying compat/002, not an invariant.

## 5. Conformance

- `TestVisionServerArgs` — the §2 table, per lineage per §4.
- `TestImageTokensForSize` — the replicated cost model for non-budgeted compat arches.
- The §3 fingerprint procedure, recorded in
  `vision-token-budget-measurements.md` (main lineage) when
  an arch is added or a payload changes.
