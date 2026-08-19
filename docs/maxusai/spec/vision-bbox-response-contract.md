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

## 0. The configuration to use

The practical distillation of everything below. Every line is measured; the
clause that establishes it is named.

```
bbox_type     "norm1000"  — pinned, AND state that the space is 1000x1000   (C1)
coordinates   named x1 / y1 / x2 / y2, never a positional array             (C2)
__IMAGE__     anchor entry, listed first, same convention as every box      (C3)
declaration   per-object                                                    (C4)
temperature   0
think         see below — permitted, per model
```

| requirement | measured |
|---|---|
| pin norm-1000 | declarations usable **21/21**, against **5/21** for free choice and **3/21** for a `real` pin |
| named coordinates | positional arrays returned `yxyx` while declaring `xyxy` in **11 of 26** cells; named fields in **0 of 13** |
| the pin across geometry | **55 of 56** cells convert 6/6 over 14 image geometries × 2 models × 2 think modes |
| per-object declaration | a measured null against document-level — 21/21 either way; preferred only because it survives truncation |

**Named coordinates carry the most weight.** Axis transposition is the one error
no numeric check can detect — a transposed box has the same range, the same
extent aspect and no scale error — and the `__IMAGE__` anchor cannot help,
because a full-image box is *symmetric under transposition*. Naming the fields
makes the error unrepresentable rather than merely discouraged. `box_2d` and
`bbox_2d` are positional and gemma-native `yxyx`; if a caller must emit them,
the order has to be fixed at that caller's own boundary, not requested.

### 0.1 think-on IS permitted for this task

**Measured 2026-08-19, under the configuration above:**

| model | think-off | think-on |
|---|---|---|
| `qwen3.8:27b-q4_K_M` | **14/14** geometries convert 6/6 | **14/14** |
| `gemma4:31b-it-q4_K_M` | **14/14** | **14/14** |
| `gemma4:26b-a4b-it-q4_K_M` (MoE) | **14/14** | **14/14** |
| `qwen3.6:35b-a3b-q4_K_M` | **14/14** | **13/14** — silent C7 failure at `sq320` only |

**111 of 112 cells** across four models. Every cell declared `norm1000` / `xyxy`
and every one was honoured — including on the gemma4 **26b** family, which is
where all 11 measured axis flips came from. With named coordinates it emitted
`xyxy` in 14/14 cells in both think modes, which is C2 working rather than luck.

[ADR 0022](../adr/0022-thinking-is-off-for-vision-work.md) turns thinking off for
vision work generally, and [ADR 0023](../adr/0023-think-mode-is-per-model-and-measured-on-policy.md)
makes the decision per model and on-policy. This section is that decision applied
to *this* task, and it is more permissive than a blanket reading of 0022 would
suggest: for a **pinned, single-image, anchored bbox request**, think-on is not
measurably worse on qwen3.8 and costs one cell out of fourteen on qwen3.6.

**What think-on is NOT safe for, on the same models:**

- **Multi-image cross-referencing.** `qwen3.6:35b-a3b-q4_K_M` think-on on
  `multi_3img` does not terminate: **122,880 tokens**, **313,054 characters** of
  reasoning, **zero** answer, byte-identical across repeats at two ceilings.
  Adding the `__IMAGE__` anchor bounds it to **10,910 tokens** and a correct
  answer, so the anchor is not optional here — it is what makes the request
  finite.
- **A `real` pin.** qwen3.6's anchor converts 6/6 at **1 of 14** geometries under
  `real`, against 14/14 under norm-1000.
- **Small square images**, for qwen3.6: the single think-on failure is `sq320`
  (320×320), which is exactly where C14 says both of C7's checks lose
  discriminating power at once.
- **Other model families.** `nemotron3` scene IoU degrades 0.840 → 0.391 under
  thinking (ADR 0022) and was not measured across geometry. The gemma4 26b axis
  flips are escaped by C2, not by thinking — see the table above; do not read
  that result as licence to send positional arrays to it.

> **RETRACTED 2026-08-19, same day it was written.** This paragraph claimed the
> norm-1000 pin terminates runaway reasoning generally, on the strength of ONE
> model. Extended to four, it does not hold — and the experiment behind it was
> confounded.
>
> | model | mode | baseline (px) | pinned |
> |---|---|---|---|
> | gemma4:26b-a4b | on | 0.334 | **0.972** |
> | qwen3.6:35b-a3b | on | 0.717 | **0.938** |
> | nemotron3:33b | on | 0.813 | **0.599** ↓ |
> | qwen3.8:27b | off | 0.977 | **0.088** ↓↓ |
>
> The `pinned` arm differed from the baseline in **two** variables, not one: it
> swapped pixels for norm-1000 *and* dropped the sentence "The image is exactly
> {w} pixels wide and {h} pixels tall". qwen3.8, given neither a dimension nor a
> convention it chose to honour, answered in **its own 2500×1400 rescale frame** —
> the same ~1.30× frame §4 measured — which `score_scene` cannot recover, hence
> 0.088.
>
> **Re-run with the dimensions retained: the claim is REFUTED, not merely
> unsupported.** Single-variable arms, 4 models × 2 think modes — the pin scores
> gemma4:26b-a4b **0.000** (full 122,880-token budget burned), qwen3.6 **0.044**
> against a **0.971** baseline, nemotron3 0.434 against 0.753. It is worse than
> the pixel baseline in 3 of 4 models and never better under thinking.
>
> What survives: gemma4:26b-a4b think-on on scene is genuinely poor (0.334 vs
> 0.973 think-off) and no prompt variant recovers it — a model property, not a
> prompt artefact. None of this bears on the bbox arms, where the norm-1000 pin
> is measured at 111 of 112 cells; `scene_single` is a different task with a
> different scorer and no anchor-derived conversion.

**The cost is tokens, not accuracy** — for qwen3.8 on this task. Budget for the
context ladder to escalate and record the rung it settles at
([SPEC H4a](vision-harness-reuse.md)); a think-on cell that caps has produced a
floor, not a cost.

### 0.2 What this configuration does NOT cover

`gemma4:31b` and `gemma4:26b-a4b` are now measured across geometry (14/14 each,
both think modes) — **but only with NAMED coordinates.** `box_2d` and `bbox_2d`
are positional and gemma-native `yxyx`, which is the exact form that produced all
11 measured flips, and no arm in this sweep requested them. **`box_2d` remains
unverified**: this result clears the models under C2, not the positional form
C2 exists to avoid.

Everything above is the synthetic six-shape scene; a photograph with ambiguous
object boundaries is a different test.

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

> **Valid JSON is not the same as usable JSON — measured 2026-08-19.** C12 covers
> a malformed box inside a well-formed response. The converse also occurs:
> `qwen3.6:35b-a3b-q8_0` think-on returned a response that `json.loads` parses
> perfectly and that simply did not contain the answers key, because the model had
> serialised that object into a **string inside an unrelated array**. The content
> was correct — `q1` and `q2` identical to the same model's passing arm — and the
> cell scored 0/3.
>
> A consumer MUST therefore distinguish *parsed* from *carries what I asked for*.
> A scorer that names the key it needs can recover the misplaced fragment; one
> that only checks `json_valid` records a model failure that did not happen. This
> is the response-level analogue of C12's principle: recover what is recoverable,
> mark it, and never let a shape error be scored as a content error. The recovery
> is implemented in `salvage.py` and reported per cell as `salvage_method`, which
> MUST be quoted with any rate that includes salvaged cells — `embedded_key` and
> `largest_object` are not the same kind of result.

## 3. Legacy responses

**C11 — Responses that predate this contract fall back to the heuristic ladder,
and their output is provisional.** The ladder (aspect test → range test →
`implied_scale` → cached per-model calibration) is specified in
[vision-bbox-coordinate-conventions.md](../vision-bbox-coordinate-conventions.md).
It is retained only for responses with no anchor. It cannot detect an axis swap
under any circumstances, and its discriminator requires a non-square image with
objects spread across the frame.

## 4. Geometry generality

**Status: measured 2026-08-19**, on two qwen families across the 14 geometries
below — [vision-campaign-2026-08-19-geometry.md](../vision-campaign-2026-08-19-geometry.md),
decision in [ADR 0030](../adr/0030-bbox-conformance-is-scoped-to-image-geometry.md).
C13–C18 were pre-registered before the run per the
[ADR 0011](../adr/0011-preflight-expectations-are-versioned-code.md) discipline;
§4.2's predictions are left as written, with outcomes appended, so a prediction
that failed stays visible rather than being quietly rewritten.

**The headline: pinning norm-1000 removes image geometry as a variable.** 53 of
54 cells convert 6/6 across every geometry, both models and both think modes.
Under a `real` pin the same models split — qwen3.8 14/14, qwen3.6 **1/14** — so
C1's pin is a correctness requirement and not only a declaration-honesty
preference. Under norm-1000 the model never has to name the frame it works in;
under `real` it must, and that is where its internal resize reaches the
coordinates.

**C13 — A conformance rate is scoped to the geometries it was measured at, and
MUST be quoted with them.** Every rate in this SPEC — the 21/21, the 11 of 26,
the 42/42, the 1 of 439, the 107 anchored cells — comes from the
`bbox_contract_*` arms. Boxes are scored in one frame only, `scene_hd` at
1920×1080, but **seven of the eight arms attach three images**, not one:
`scene_hd` (1920×1080) together with `document.png` (**1568×1568**, square) and
`chart.png` (**1280×960**, 4:3). The encoder has therefore never been shown a
lone HD image in any rate-producing cell, and it has been shown a square image
throughout. What is unmeasured is a *scored* geometry other than 16:9 HD; what
is uncontrolled is the attached distractors' geometry.

No rate here is established as a property of the contract. Each is a property of
the contract *scored at 16:9 HD, in a three-image request whose other two images
were 1568×1568 and 1280×960* — a description no rate in this document currently
carries. A consumer converting a single pasted screenshot matches none of those
conditions, and the SPEC previously admitted this only in a closing footnote.

**C14 — C7's discriminating power is a joint property of the image geometry and
where the objects sit in it, and MUST be reported alongside any `self_check`
rate.** The two checks C7 requires do not degrade together, and each has a
condition that silences it. Both conditions are computable **from the response
alone** — this clause must not reach for `W` and `H`, because C7's whole value
is that it "uses ONLY the response … not even the image dimensions", and a
validator that needs the image dimensions cannot run on the unmeasured image it
exists to serve:

| condition (response-only) | effect on C7 |
|---|---|
| object extent aspect `ew/eh` ∈ **0.8–1.25** | **aspect check non-discriminating** — `bbox_self_check` tests `(aw/ah)/(ew/eh)` against exactly that band, so a fabricated *square* anchor lands inside it and is indistinguishable from a correct one |
| anchor implies `real` with `max(X, Y) ≤ 1000` | **range check cannot separate `real` from `norm1000`** — real coordinates are ≤1000 by construction and pass a norm-1000 range test silently |
| both at once | **C7 has no discriminating power.** Anchor extent *magnitude* is the only remaining signal |

> **Measured 2026-08-19.** C7's silent-failure rate is geometry-dependent, and
> the SPEC's 1-in-107 figure describes 16:9 HD rather than the validator. Under
> a `real` pin across 14 geometries, qwen3.6 produced **four silent failures in
> fourteen cells** — `vga`, `paste2`, `paste4`, `paste6`, each passing
> `self_check` while the anchor converted only 2–3 boxes of 6. The mechanism is
> the known one; geometry changes how often it fires. Quote C7's failure rate
> with the geometry it was measured at, or not at all.

The band is **0.8–1.25**, taken from `bbox_self_check` itself, not a tighter
near-square guess: it is roughly five times wider than "aspect ≈ 1" and so
catches far more images than an intuition about squareness suggests. Note the
check compares the anchor against the **objects' extent**, not against the
image — which is why the C7 amendment's false reject was caused by objects
clustering rather than by the image's shape.

A 320×320 image satisfies both rows simultaneously. This is the case the
contract is least able to defend and has never been measured; it is also an
entirely ordinary thing to paste into a chat window. Near-squareness is enough —
it does not require exact 1:1, and it is not confined to small images.

**C15 — Alignment-sensitive geometries MUST be derived from the
architecture's `patch_stride`, never hardcoded.** Alignment is a joint property
of the image and the encoder, not of the image alone. The corpus has two
strides: 32 (`nemotron_h_omni`, `qwen35`, `qwen35moe` — patch 16 × merge 2) and
48 (`gemma4` — patch 16 × merge 3), both recorded per-arch in
`vision-suite/preflight/expectations.toml`.

**1920×1080 is misaligned on the height axis for every architecture in the
corpus**: `1080/32 = 33.75` and `1080/48 = 22.5`. The scored fixture has
therefore never been on a clean grid — `scene_hd` pads to 1920×1088 (+0.7%) at
stride 32 and to 1920×1104 (+2.2%) at stride 48. The pad is **not uniform across
a request**: the two attached distractors are exactly grid-aligned at stride 32
(`1568/32 = 49`; `1280/32 = 40`, `960/32 = 30`) and carry their own, different
pads at stride 48, so a request has always mixed padded and unpadded images. A frame the model reports as
~1.30× its input may be resize *and* pad, and no measurement to date separates
them. An aligned twin must consequently be chosen by the arch's stride: 1920×1088 at
stride 32, 1920×1104 at stride 48. **Both are generated as ordinary fixtures and
both are literals** — C15 constrains the *selection*, not the rendering, since a
fixture is a PNG written ahead of time while the architecture is known only once
a model is loaded. A run that pairs `hd` with the wrong twin measures a 1.4% pad
against a 0.7% one and reports the difference as a model property.

**C16 — The geometry set MUST include non-round sizes.** Round dimensions
cluster on or near the patch grid and systematically under-sample the
misalignment the contract will actually meet. Images are pasted into a chat
window at whatever size they happen to be — a cropped screenshot is 1387×907,
not 1920×1080 — so a set built only from round numbers measures the friendliest
corner of the input space and reports it as the contract's behaviour.

**C17 — Frame direction MUST be measured in both directions.**

> **Measured 2026-08-19, and the original text below was wrong.** Frames
> *smaller* than the input are not rare — they are the common case once the
> input stops being 1920×1080: **0.46×–0.96×** on qwen3.6 and **0.67×–0.93×** on
> qwen3.8, across 14 geometries. The reported frame is also not a smooth
> function of input size: qwen3.8 returns `2560×1440` for four different inputs
> and `2337×1754` for two more, snapping to canonical sizes rather than scaling.
> That is the concrete reason a caller cannot infer the frame and must read it
> from the anchor. The claim below was an artefact of only ever sending one
> geometry.

Every anchor that has been *usable* reports a frame *larger* than the input. The
one observed frame smaller than the input is the fabricated `real/[1200, 900]`
cell in the C7 amendment above — on a 1920×1080 input, passing **both** checks
with `hits_anchor` 3 against a best-fit of 6. The usable observations are all one
way: qwen3.8 returns
~2496 on a 1920-wide image, and the `ref_size` values in C5 (2500×1406,
2560×1440, 2324×1312) are all above the input width. The conversion in C10 has
therefore only ever been exercised where the frame exceeds the image. The
stride-32 frame ceiling is `budget_max_tokens = 4096` × 32² = 4,194,304 px, or
~2731×1536 at 16:9, so an input above ~2731 wide inverts the ratio and tests the
opposite sign.

**C18 — The image-token budget configuration MUST be recorded with every
geometry result.** `gemma4` on fork defaults (40…1120) produces a saturating
token curve; the same model pinned at 1088/1120 produces a flat one. A frame
that is constant across geometries therefore means "budget ceiling reached"
under one configuration and "pinned, as instructed" under the other, and the
result cannot be read without knowing which. This is the geometry axis's
equivalent of [ADR 0012](../adr/0012-benchmark-report-templates.md) rule 6,
which requires `num_ctx` in the cell for exactly the same reason: a number whose
meaning depends on an unrecorded setting is not a measurement.

### 4.1 The geometry set

Two tiers. Tier 1 isolates one variable per pair; tier 2 measures reliability
under the distribution the contract actually meets. Regimes are computed from
`image_min_pixels` and `budget_max_tokens` in `expectations.toml`, not assumed.

**Image count is held at one.** The geometry axis runs on
`bbox_contract_anchored_1img` — the `bbox_contract_anchored` prompt sent with
one image instead of three — **not** on the seven three-image arms that produced
the rates in §1–§3. That arm had to be added: neither existing arm can carry the
axis. `bbox_contract` is single-image but requests no `__IMAGE__` entry, so it
reports no frame at all and every prediction in §4.2 is about the reported
frame; `bbox_contract_anchored` has the anchor but sends three images. Varying geometry inside
a three-image request would move image count, total token load and the
distractors' own geometry simultaneously, and the resulting cell could not
attribute anything to the scored image's shape — the confound C13 exists to
prevent. The cost is deliberate and must be stated in the report: **geometry
results are not comparable to the distractor-condition rates in §1–§3.** The
`hd` cell establishes the single-image baseline that the rest of the set is read
against, and comparing the two conditions at `hd` also finally measures the
distractor effect itself, which no cell has ever isolated.

**Tier 1 — controlled pairs.**

| id | W×H | aspect | isolates | qwen35 | gemma4 |
|---|---|---|---|---|---|
| `hd` | 1920×1080 | 1.778 | control — every existing rate in this SPEC | in-range | in-range |
| `hd_al32` | 1920×1088 | 1.765 | **alignment only**, stride-32 arches (C15) | in-range | in-range |
| `hd_al48` | 1920×1104 | 1.739 | **alignment only**, stride-48 arches (C15) | in-range | in-range |
| `sq320` | 320×320 | 1.000 | **both C14 rows at once**; aligned @32, +10.3% pad @48 | UPscale | UPscale |
| `vga` | 800×600 | 1.333 | C14 range row; asymmetric pad (+6.1% @48) is aspect-visible | UPscale | in-range |
| `portrait` | 1080×1920 | 0.562 | **aspect only** — identical pixel count to `hd` | in-range | in-range |
| `uhd` | 3072×1728 | 1.778 | above budget; **aligned at both strides** | DOWNscale | DOWNscale |
| `uhd4k` | 3840×2160 | 1.778 | **C17 frame < input**; misaligned @32, aligned @48 | DOWNscale | DOWNscale |

The set spans all three resize regimes — below floor, in range, above budget —
for both architecture families. The regime columns are *derived* from
`image_min_pixels` and `budget_max_tokens`, but they are consistent with the
token curves already measured on the standard ladder
([vision-token-budget-measurements.md](../vision-token-budget-measurements.md)):
qwen35 reads 1038 / 1026 / 2042 / 2403 / 4058 — floored at the fixed
`--image-min-tokens 1024` for the two smallest geometries, then scaling, then
pressed just under the 4096 cap at 3072×1728, which is the downscale the table
predicts.

**One consequence is a free control.** On qwen35, `sq320` and `vga` are *both*
below the 1024-token floor, so they arrive at the encoder with the **same token
count and different aspect** (1.000 against 1.333). Any difference in reported
extent between those two cells cannot be a token-count effect, which makes them
a second, independent read on P3 that costs nothing extra to run.

**Tier 2 — pasted sizes.** Drawn once from `random.seed(20260818)`, constrained
to aspect 0.4–3.0 and rejected if divisible by 32 or 48 on either axis, then
frozen as literals so the set is reproducible without re-running a generator:

| id | W×H | aspect | pad @32 | pad @48 | C7 aspect | C7 range |
|---|---|---|---|---|---|---|
| `paste1` | 1668×733 | 2.276 | +2.1% | +5.5% | ok | ok |
| `paste2` | 2812×2135 | 1.317 | +0.6% | +1.9% | ok — by 5% | ok |
| `paste3` | 1235×1181 | 1.046 | +1.3% | +2.7% | **DEAD** | ok |
| `paste4` | 2750×2379 | 1.156 | +1.0% | +2.1% | **DEAD** | ok |
| `paste5` | 3030×1549 | 1.956 | +1.6% | +3.7% | ok | ok |
| `paste6` | 3011×2317 | 1.300 | +1.8% | +1.9% | ok — by 4% |ok |

Scored against C7's real 0.8–1.25 band (using image aspect as a stand-in for
extent aspect, which holds for this fixture because its shapes span the frame),
**two of six random pastes fall in the dead band and two more clear it by under
5%**. Neither dead one is small: `paste3` is 1235×1181 and `paste4` is
2750×2379. The blind spot is not a small-image problem, and it is not rare —
which a set of round 16:9 sizes would never have shown.

### 4.2 Pre-registered predictions

Stated before measurement so they can fail.

- **P1 — token-grid frame.** If the reported frame tracks the token grid rather
  than the input, the extent should follow the arch's *token* curve, not its
  pixel dimensions. For `gemma4` on fork defaults that curve **saturates rather
  than being flat** — measured at 132 / 363 / 922 / 1091 / 1082 tokens across
  the standard ladder geometries
  ([vision-token-budget-measurements.md](../vision-token-budget-measurements.md))
  — so the prediction is a *rising* extent across `sq320` and `vga` and a
  *constant* one across `hd`, `uhd` and `uhd4k`, with the knee near the 1120
  ceiling. A uniformly flat extent would refute P1 as surely as a uniformly
  rising one. If P1 holds it explains why the frame artefact has only ever been
  seen on qwen.

  > **Do not restate this as "gemma4 is flat".** It is flat only when *pinned*
  > (1133 / 1091 / 1102 / 1091 / 1082 at 1088/1120), and the geometry run must
  > record which configuration it used — see C18. The same model is flat or
  > saturating depending on a knob, and reading a pinned result as a default one
  > inverts the conclusion.
- **P2 — padding is in the frame.** For geometries misaligned to the arch's
  stride, the reported extent's aspect differs from the input aspect by the pad
  fraction. Detectable as `hd` vs `hd_aligned`. **This one may be
  unresolvable**: the HD pad is 0.74% at stride 32, ≈7 units in norm-1000, at
  the edge of what the space can express. `vga` at stride 48 (1.9% aspect shift,
  ≈19 units) is the sensitive probe; `hd` is the one we care about.
- **P3 — aspect-preserving vs letterboxed.** At `portrait`, an aspect-preserving
  norm-1000 yields an extent aspect ≈0.562; a square-letterboxed space yields
  ≈1.0. `hd` alone cannot distinguish these two hypotheses, which is why this
  pair exists.
- **P4 — square hides its own pad.** At `sq320` both axes pad equally, so the
  extent aspect stays 1.0 regardless of padding. If the pad is in the frame it
  is visible only in extent magnitude (336 vs 320 at stride 48), never in
  aspect. A validator relying on aspect alone cannot see it.

### 4.3 Acceptance

The contract **holds at geometry G** for a model iff, at G: the `__IMAGE__`
anchor is present and well-formed (C3, C12); the anchor-derived space converts
to pixels within the same IoU tolerance as the `hd` baseline for that model; and
where C14 marks a check non-discriminating, the result is recorded as
*undetermined* rather than as a pass. **A `self_check` pass on a geometry where
C14 says the check has no power is not evidence of conformance**, and pooling it
into an aggregate rate is the specific error C13 exists to prevent.

Scoring is restricted to shape IoU and anchor metrics. Text-dependent metrics —
labels, the tiny serial — are **not** comparable across this set: glyph size
scales with the fixture, a 14 px serial becomes ~2 px at 320×320, and a failure
there is a legibility failure being misread as a contract failure. Text metrics
MAY be recorded as diagnostics; they MUST NOT gate conformance at any geometry
other than `hd`.

### 4.4 Measurement order

`qwen35` / `qwen35moe` first — qwen3.6 and qwen3.8 are the family where the
rescaled frame is already known to occur (C5: `ref_size` 2500×1406, 2560×1440,
2324×1312 on one 1920×1080 input), so they carry the most signal per cell and
will show geometry instability soonest if it exists.

**`gemma4` is the control and MUST still be measured.** Its stability across
geometries is not a premise of this section — it *is* P1, and P1 is a
prediction. Treating "gemma4 will be stable" as an assumption and skipping it
would leave the qwen result with nothing to contrast against: a qwen extent that
moves with input size means one thing if gemma4's holds constant and something
entirely different if both move. The control is where P1's explanatory claim —
that budget-filling is *why* the artefact is qwen-only — either survives or
dies.

Order within the run: `hd` first at every model, as the tie to the existing
corpus, then tier 1, then tier 2. Both think modes throughout, per
[ADR 0023](../adr/0023-think-mode-is-per-model-and-measured-on-policy.md) and the standing rule that a
think-on cell is not optional; the context ladder runs per SPEC H4a and the rung
reached is a first-class result, not a diagnostic.

## 5. Conformance

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
| C13 | `geometry` / `image_size` / `image_aspect` / `label_px_clamped` on every cell when the axis is in use; `summarize_geometry.py` renders them. Measured over 14 geometries × 2 models. The `hd` cell of the §4.1 set is the tie-back, but only to the single-image `bbox_contract` arm — it **cannot** reproduce the §1–§3 rates, which are three-image distractor-condition cells (`vision_suite.py:1199` ff.) and differ in image count, not just geometry |
| C14 | **Enforcement pending.** `bbox_self_check()` returns a bare pass/fail and cannot express "non-discriminating". It needs a third state keyed off the **objects' extent aspect** and the anchor's implied frame — both already computed inside the function — so that it stays inside C7's response-only invariant. Keying it off `W`/`H` would break that invariant and is forbidden |
| C15 | `patch_stride` per arch in `vision-suite/preflight/expectations.toml` — already versioned per ADR 0011. The `hd_aligned` fixture MUST read it rather than restating it |
| C16 | The §4.1 tier-2 literals, frozen from `random.seed(20260818)`. Enforced by their being literals: a regenerated set would silently change what was measured |
| C17 | `bbox_contract_real_1img` — the only arm that forces a model to name its frame. Measured: derived frames below input are the common case (0.46×–0.96× qwen3.6, 0.67×–0.93× qwen3.8), settling the direction the row below called unobserved. The direction is not wholly unobserved — the fabricated `real/[1200, 900]` cell in the C7 amendment went that way and passed both checks — which is the argument for measuring it, not against |

All rates: server `0.32.5-maxusai-a5d65906`, powermode 2, think-off per
[ADR 0022](../adr/0022-thinking-is-off-for-vision-work.md), cold restart per
model, 7 configurations × 3 repeats.

**Not covered.** One fixture at one image size, 1920×1080. A square image, or
one where `max(W, H) ≤ 1000`, makes the C7 aspect check non-discriminating and
the range check ambiguous between `real` and `norm1000`. **§4 promotes this
footnote to normative clauses C13–C17** and specifies the geometry set that
measures it; until that run lands, every rate above should be read as scoped to
16:9 HD. C7's thresholds (2% range tolerance, 0.8–1.25 aspect band) separate
this corpus cleanly but are not derived from a model of the error distribution.

