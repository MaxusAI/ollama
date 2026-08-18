# SPEC: vision bounding-box response contract

MaxusAI-fork specification. Status: **implemented in the benchmark harness**
(`docs/maxusai/vision-suite/vision_suite.py` — eight probes and the
`bbox_self_check` validator). **Not enforced in server code**: this is a
contract between a caller and a vision model, not a property of the runner, so
nothing in `server/` or `llm/` validates it. Written 2026-08-16.

Normative contract for how a bounding-box request is phrased and how the
response is converted to pixels. The decision and its rationale are
[ADR 0027](../adr/0027-bbox-requests-pin-norm1000-and-carry-an-anchor.md); the
per-model measurements and the legacy recovery ladder are in
[vision-bbox-coordinate-conventions.md](../vision-bbox-coordinate-conventions.md);
the campaign is
[vision-campaign-2026-08-16-seven-model.md](../vision-campaign-2026-08-16-seven-model.md).

**The problem this exists to solve.** Every model in the corpus locates shapes
accurately and describes the convention it used unreliably. Grounding is not the
variable — `hits_bestfit` is 6/6 in nearly every failing cell — so a consumer's
whole risk is converting correct coordinates with the wrong rule.

## 1. The request

**C1 — Requests MUST pin norm-1000, and MUST state what the space is.** The
convention is not the caller's stylistic choice; it is load-bearing, and the
wrong pin is worse than no pin. Measured over 7 configurations × 3 repeats,
distractor condition, think-off:

| pinned convention | declarations usable |
|---|---|
| **norm-1000** | **21/21** |
| `norm1` (0.0–1.0) | 15/21 — gemma4 emits norm-1000 anyway, both engines |
| `real` pixels | **3/21** |
| *(none — free choice)* | 5/21 |

The required wording, or a paraphrase preserving both clauses:

> Use `"bbox_type": "norm1000"` — each axis scaled independently to 0-1000, x by
> 1000/width and y by 1000/height. The coordinate space is 1000x1000 whatever
> the image's shape is.

Both clauses are required. Free choice scores 5/21; adding the pin *and* the
statement of the space reaches 21/21. Do not shorten this to "use norm-1000".

> **C1 fixes the type, not the order — and the 21/21 does not generalise past
> 31b.** The 21/21 above was measured on `gemma4:31b`, `qwen3.6:35b`,
> `qwen3.8:27b` and `nemotron3:33b`. On **every `gemma4:26b` variant** — a4b-MoE,
> nvfp4, mxfp8 and mlx-bf16 alike — a pinned request using a *positional array*
> comes back `norm1000/`**`yxyx`** while declaring `xyxy`, scoring 0/6. The
> pinned type is obeyed; the axis order is not, and C1 has no power over it.
> **C2 is what fixes the order**, and the separation is clean: across the models
> measured so far, a positional array emits `yxyx` while declaring `xyxy` in
> **11 of 26** cells, and named coordinates in **0 of 13**. Treat C1 and C2 as
> independent requirements, not belt-and-braces — dropping C2 because "we pinned
> the convention" reintroduces the one error no numeric check can detect.
>
> Interim, from the 18-model × both-think-modes run of 2026-08-16 (n=1 per cell,
> run still in progress). The direction is not in doubt — 0/13 against 11/26 —
> but the exact rates will be restated when the run completes.

**C2 — Coordinates MUST be named fields (`x1`, `y1`, `x2`, `y2`), never a
positional array.** A positional array is the only reason a `coord_order` field
has to exist, and axis order is the one error no numeric check can detect: a
transposed box in a normalized space has the same range, the same extent aspect
and no scale error. Naming the fields makes the transposition unrepresentable
rather than discouraged.

This is the requirement with the largest measured effect, and it is **not**
subsumed by C1 (see the note there). Across the models measured so far:

| coordinate form | emitted `yxyx` while declaring `xyxy` |
|---|---|
| positional array (`bbox_contract_pinned`, `bbox_contract_perobject`) | **11 of 26** |
| named fields (`bbox_contract_anchored`) | **0 of 13** |

Every flip is a `gemma4` cell, but it spans sizes (12b and 26b), engines (GGUF
and MLX) and quantisations (q4_K_M, nvfp4, mxfp8, bf16), so it is a property of
the family rather than of one build. It is also **not** reliably escaped by
thinking: three of the four `gemma4:26b` variants stop flipping with think-on,
but `gemma4:26b-a4b-it-q4_K_M` flips in **both** modes.

**C3 — Requests MUST require an `__IMAGE__` entry** covering the whole image,
in the same convention as every other entry, listed first.

**C4 — Declaration placement is free.** `bbox_type` MAY be declared once at the
document level or repeated on each object; both measured 21/21 with identical
values. Per-object is the marginally safer default because it survives
truncation. Implementations MUST accept either, and a document-level declaration
takes precedence when both are present.

Conforming request shape:

```json
{"objects": [{"label": "__IMAGE__", "bbox_type": "norm1000",
              "x1": 0, "y1": 0, "x2": 1000, "y2": 1000},
             {"label": "…", "bbox_type": "norm1000",
              "x1": 72, "y1": 148, "x2": 219, "y2": 333}]}
```

## 2. Converting the response

**C5 — The coordinate space MUST be derived from the `__IMAGE__` anchor, and a
declared `ref_size` MUST NOT be trusted.**

| anchor returns | space is |
|---|---|
| `[0, 0, ~1, ~1]` | `norm1` |
| `[0, 0, ~1000, ~1000]` | `norm1000` |
| `[0, 0, X, Y]` | `real`, in a frame of **X × Y** |

`ref_size` is unreliable in both directions: qwen3.8 GGUF reports 2500×1406,
2560×1440 and 2324×1312 across runs on one 1920×1080 input, and nemotron3 omits
it entirely while declaring `real`. The third row is why the anchor exists — it
recovers a frame no dialect search can guess (qwen3.8, `ref_size` absent: anchor
derives `[2338, 1316]` for 6/6, against a best-fit of **1/6**).

**C6 — An anchor MUST be validated before use. It is not automatically
trustworthy.** A model that is not actually working in the requested space can
answer "where is the whole image" from **semantic knowledge of the image
dimensions** rather than by emitting a box in its working space, at which point
the anchor is a second copy of the false declaration. Measured across the two
adversarial arms (42 responses, 15 genuinely mis-declared):

| outcome | cells |
|---|---|
| anchor recovers what the declaration could not | **12/42** |
| anchor and declaration both already correct | 15/42 |
| anchor inherits the declaration's error | **9/42** |
| anchor invents a third, also-wrong frame | **6/42** |

**C7 — Validation is two checks, and both are required.** Neither is sufficient
alone; each catches exactly what the other is blind to.

- **Range** — every coordinate MUST fit the space the anchor implies (`norm1`
  ⇒ ≤ 1.0; `norm1000` ⇒ ≤ 1000; `real` ⇒ ≤ the derived frame), within 2% for
  edge rounding. Catches a pure **scale** lie: gemma4 declares `norm1` and emits
  norm-1000, and since both spaces are square no shape test can see it. **6 of
  the 15**, caught only here.
- **Aspect** — the anchor's own aspect ratio MUST match the objects' extent
  aspect, within a factor of 0.8–1.25. Catches a **fabricated frame**: asked for
  `real`, gemma4 MLX returns `[0, 0, 1920, 1080]` — the true image size,
  answered from knowledge — while its boxes are norm-1000, so the anchor reads
  1.778 against an extent of 1.099. **9 of the 15**, caught only here.

Both tests use the response alone: no ground truth, no image content, not even
the image dimensions. They therefore run unchanged on an image nobody has
measured, which is the point.

> **The 42/42 separation is a property of the original 42 responses, not of the
> mechanism.** Re-measured over 107 anchored cells in the
> [18-model campaign](../vision-campaign-2026-08-17-eighteen-model.md), C7 has
> **one silent failure and one false reject**:
>
> - **Silent failure** — `qwen3.6:35b-a3b-q4_K_M` think-on `adv_real`: the anchor
>   claimed `real/[1200, 900]`, a frame that was never sent, and **both range and
>   aspect passed** while `hits_anchor` was 3 against a `hits_bestfit` of 6. A
>   fabricated frame whose aspect happens to match the objects' extent defeats
>   both checks. This is the known gap.
> - **False reject** — `nemotron3:33b-bf16` think-on `adv_real`: rejected on
>   aspect (anchor 1.00 vs extent 1.43) while the anchor was in fact correct. The
>   aspect test assumes the objects span the frame; when they cluster, a correct
>   normalized anchor looks inconsistent with them.
>
> Treat C7 as a strong filter, not a proof. It remains worth running — it caught
> 14 of 16 genuinely unusable anchors — but a passing `self_check` is not a
> guarantee, and a failing one on a sparse scene deserves a second look.

**C8 — A response failing either check MUST be rejected, not converted.**
Rejection MUST NOT fall back to a best-fit dialect search (C9). A caller with no
ground truth cannot verify a best-fit result, and it fails hardest where it is
most needed.

**C9 — Best-fit dialect search is a diagnostic, never a consumer path.**
`hits_bestfit` exists to quantify the cost of a wrong declaration. It silently
rescues models that describe their own output incorrectly, which is how this
class of error stayed hidden, and it scores **1/6** on qwen3.8's rescaled frame
where the anchor scores 6/6. A failing probe cell MUST NOT be "fixed" by
relaxing a scorer to best-fit.

**C10 — Conversion.** Having derived space *S* and passed C7, convert to pixels
of the image as sent: `norm1000` ⇒ `x·W/1000`, `y·H/1000`; `norm1` ⇒ `x·W`,
`y·H`; `real` in frame `X×Y` ⇒ `x·W/X`, `y·H/Y`.

**C12 — Boxes MUST be well-formed, and a malformed box MUST be dropped
individually rather than invalidating the response.** Every box must satisfy
`x1 < x2` and `y1 < y2` **in the order its declaration implies**. A response
carrying one malformed box among five good ones is not a C8 rejection: the
coordinate space is fine and only that box is unusable.

The evaluation order is load-bearing. Testing raw coordinates would flag every
legitimately transposed box as degenerate, and gemma4 emits `yxyx` across four
quantisations — a large fraction of a perfectly good corpus would be discarded.
Transpose first, then test.

Measured across 439 contract responses on this host, exactly **one** box is
degenerate: `gemma4:26b-mxfp8` think-off returned `ANCHOR` as
`x1=0.74, x2=0.215`, a digit slip for `0.074`, inside an otherwise correct
`norm1/xyxy` response. `hits_bestfit` was 5 as well, so no dialect recovers it —
it is one bad box, not a bad convention, and that is precisely the distinction
C12 exists to draw.

This requirement deliberately does **not** feed `self_check` (C6/C7). That gates
the coordinate space for the whole response and its remedy is wholesale
rejection; conflating the two would discard five good boxes over one digit slip.

## 3. Legacy responses

**C11 — Responses that predate this contract fall back to the heuristic ladder,
and their output is provisional.** The ladder (aspect test → range test →
`implied_scale` → cached per-model calibration) is specified in
[vision-bbox-coordinate-conventions.md](../vision-bbox-coordinate-conventions.md).
It is retained only for responses with no anchor. It cannot detect an axis swap
under any circumstances, and its discriminator requires a non-square image with
objects spread across the frame.

## 4. Conformance

| requirement | enforced by |
|---|---|
| C1 | `bbox_contract_pinned`, `bbox_contract_perobject` (21/21); `bbox_contract_adv_real` (3/21) and `bbox_contract_adv_norm1` (15/21) establish that the pin choice is not free; `bbox_contract_multi` (5/21) is the free-choice control |
| C2, C3 | `bbox_contract_anchored` — 21/21 named coordinates used verbatim, 21/21 anchors at exactly `[0,0,1000,1000]` |
| C4 | `bbox_contract_pinned` vs `bbox_contract_perobject`, 21/21 each with identical declarations — a measured null |
| C5 | `anchor_implied_type` / `anchor_implied_ref` / `hits_anchor` on every contract score |
| C6, C7 | `bbox_self_check()`; separated usable from unusable anchors **42/42** across the original adversarial arms — 27 accepted and all correct, 15 rejected and all genuinely bad. **Superseded**: over 107 anchored cells it is one silent failure and one false reject — see the C7 amendment above, which this row previously contradicted |
| C8, C9 | `self_check` recorded next to `hits_anchor` precisely so `self_check == true` with `hits_anchor < 6` is visible; that pairing is the signature that would falsify C7 |
| C10 | `factors()` in `score_bbox_contract` — the same three conversions, applied per box under the declared or anchor-derived space |
| C12 | `degenerate_boxes` / `degenerate_labels` on every contract score, evaluated in the declared order. 1 of 439 responses on this host; verified not to fire on any of the 11 `yxyx`-transposed cells |
| C11 | **Nothing enforces this.** The ladder is documented prose, exercised only by the `implied_scale` / `iou_at_implied_scale` diagnostics. Legacy handling is best-effort by construction |

All rates: server `0.32.5-maxusai-a5d65906`, powermode 2, think-off per
[ADR 0022](../adr/0022-thinking-is-off-for-vision-work.md), cold restart per
model, 7 configurations × 3 repeats.

**Not covered.** One fixture at one image size. A square image, or one where
`max(W, H) ≤ 1000`, makes the C7 aspect check non-discriminating and the range
check ambiguous between `real` and `norm1000`; neither is measured. C7's
thresholds (2% range tolerance, 0.8–1.25 aspect band) separate this corpus
cleanly but are not derived from a model of the error distribution.
