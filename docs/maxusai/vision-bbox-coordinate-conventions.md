# Bounding-box coordinate conventions, measured per model

MaxusAI-fork reference. Written 2026-08-16 after a bbox result was twice
misread as a grounding failure when it was a coordinate-frame error.

**The one thing to take away:** every vision model in this corpus locates shapes
accurately, and they report those locations in **four different conventions**.
None of them matched the "absolute pixels, as asked" that the suite's prompts
assumed. A scorer that fixes the convention in advance measures dialect
compliance and calls it vision.

## The conventions that exist

| source | field | order | space |
| --- | --- | --- | --- |
| Gemma 3/4, Gemini, PaliGemma (documented) | `box_2d` | `[ymin, xmin, ymax, xmax]` | normalized 0–1000 |
| Qwen2.5-VL / Qwen3-VL (documented) | `bbox_2d` | `[x1, y1, x2, y2]` | absolute pixels **of the resized image** |
| [ms-swift](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md) training format | `objects.bbox` | — | `bbox_type`: `real` \| `norm1`, auto-converted to norm-1000 for qwen3-vl |

`bbox_type` is why this file uses ms-swift's vocabulary rather than inventing
one: a response that satisfies the contract is directly usable as fine-tuning
data.

## Measured, this corpus

`bbox_contract`, think-off, single 1920×1080 image, six labelled shapes, from
the seven-model sweep in
[vision-campaign-2026-08-16-seven-model.md](vision-campaign-2026-08-16-seven-model.md)
— server `0.32.5-maxusai-a5d65906`, powermode 2, one provenance for every row.

| model | engine | declared | `ref_size` | hits | IoU | contract |
| --- | --- | --- | --- | --- | --- | --- |
| gemma4:31b-it-q4_K_M | GGUF | `norm1000/xyxy` | — | 6/6 | 0.961 | ✅ |
| gemma4:31b-nvfp4 | MLX | `norm1000/xyxy` | — | 6/6 | 0.962 | ✅ |
| qwen3.6:35b-a3b-q4_K_M | GGUF | `real/xyxy` | [1920, 1080] | **1/6** | 0.045 | ❌ |
| qwen3.6:35b-a3b-nvfp4 | MLX | `norm1000/xyxy` | [1920, 1080] | 6/6 | 0.957 | ✅ |
| qwen3.8:27b-q4_K_M | GGUF | `real/xyxy` | **[2500, 1406]** | 6/6 | 0.971 | ✅ |
| qwen3.8:27b-nvfp4 | MLX | `norm1/xyxy` | [1000, 1000] | 6/6 | **0.990** | ✅ |
| nemotron3:33b-q4_K_M | GGUF | `real/xyxy` | **missing** | **0/6** | — | ❌ |

Five of seven are honest, across **four** conventions. Every one of the seven
located all six shapes; both failures score `hits_bestfit` 6/6.

The two failures share a direction: **declare `real`, emit normalized**. That is
the direction that silently halves every box, and it is the one a fixed-dialect
scorer reports as a grounding miss.

### qwen3.8 GGUF reports in a 1.302× frame, and will say so

It declares `ref_size [2500, 1406]` for a 1920×1080 input — 1.302 on both axes.
That is the same factor that made its `multi_3img` DYNAMO box look like a 250px
miss: raw IoU **0.079**, but **0.909** once divided out. The shape was always
found; only the frame was wrong.

This is why `ref_size` is mandatory for `real` rather than optional. "Absolute
pixels" is not a convention — it is a convention *plus a frame*, and Qwen-VL's
documented behaviour (coordinates relative to the resized image) means the frame
is routinely not the one the caller sent.

`ref_size` is also the only thing that recovers this row. Best-fit search over
type × order scores **1/6** on it, because no search over dialects can guess a
1.3× frame. The declaration outperforms the search.

### gemma4's axis order is condition-dependent, and it does not say when it flips

Gemma's `box_2d` is documented as `[ymin, xmin, ymax, xmax]`. On the
single-image probe, both engines emit `xyxy` and declare `xyxy` — honest, and
contrary to the documentation. Attach distractor images and the *same model*
emits the same numbers transposed, while `coord_order` still reads `xyxy`:

| label | `bbox_contract` | `bbox_contract_multi` |
| --- | --- | --- |
| ANCHOR | `[72, 147, 220, 335]` | `[146, 72, 334, 221]` |
| BEACON | `[321, 108, 470, 304]` | `[107, 321, 303, 469]` |
| DYNAMO | `[114, 555, 251, 796]` | `[555, 114, 795, 251]` |

Best-fit resolves the right column as `norm1000/yxyx` at 6/6; scored in the
declared dialect it is 0/6, IoU 0.044.

So the documented `yxyx` is real, but it is a **mode the model slips into**, not
a fixed property — and the declaration does not track it. Do not hard-code
either order, and do not assume a declaration verified on one prompt shape holds
on another.

**This is the worst failure in the corpus, because none of the recovery steps
below catch it.** The aspect test discriminates normalized from real, not `xyxy`
from `yxyx`: in a normalized space both axes span 0–1000 whichever order they
are written in, so the extent looks identical. The range test sees nothing. The
`implied_scale` diagnostic sees no scale error, because there isn't one. A
transposed box is only detectable against ground truth the consumer does not
have. Treat axis order as requiring per-model *and per-prompt-shape*
calibration — step 4 below is the only defence.

## Determining the format on an image you cannot check

The normative answer is a SPEC, not this file:
**[SPEC: vision bounding-box response contract](spec/vision-bbox-response-contract.md)**
(C1–C11), decided in
[ADR 0027](adr/0027-bbox-requests-pin-norm1000-and-carry-an-anchor.md).

In one line: **pin norm-1000 and say what the space is, name the coordinates,
require an `__IMAGE__` anchor, derive the space from that anchor rather than
from any `ref_size`, validate it with the range and aspect checks, and reject
rather than convert on failure.**

This file is the *measurement record* behind that contract — what each model
actually does, and how to cope with responses that predate it. The headline
numbers it rests on, all 7 configurations × 3 repeats:

| condition | declarations usable |
| --- | --- |
| free choice of convention | 5/21 |
| pinned norm-1000 | **21/21** |
| pinned `norm1` | 15/21 |
| pinned `real` | 3/21 |
| anchor alone, adversarial arms | 27/42 |
| anchor + range/aspect validation | **42/42 correctly classified** |

Three findings from that record are worth carrying in the reader's head, because
each one broke an assumption:

**norm-1000 divides each axis by its own dimension**, so the space is square and
does **not** preserve aspect ratio. Scaling both axes by `1000/W` is wrong and
produces a plausible-looking result. This is the most commonly got-wrong part of
the convention, and stating it explicitly in the prompt is what moves 5/21 to
21/21 — the pin alone does not.

**Where the declaration sits is irrelevant.** Top-level and per-object are 21/21
each. An earlier ad-hoc run suggested per-object rescued top-level 3/3 vs 0/3;
it does not reproduce, because that prompt lacked the sentence above.

**An anchor can inherit the lie it exists to catch.** Asked "where is the whole
image", a model not actually working in the requested space can answer from
semantic knowledge of the image dimensions instead of emitting a box in its
working space. gemma4 returns `[0,0,1920,1080]` — correct about the image,
useless as calibration — while its boxes are norm-1000. That is why C6/C7
mandate validation rather than trust, and why the anchor alone is 27/42 while
the validated anchor is 42/42.

## Recovering coordinates from nemotron3 (and qwen3.6 GGUF)

nemotron gets the declaration wrong in the dangerous direction: it declares
`real`, omits the required `ref_size`, and its boxes are actually **norm-1000**.
`hits_bestfit` is 6/6 on `norm1000/xyxy` — it **located every shape** and
described the convention wrongly. A consumer trusting the declaration would
place every box at roughly half scale.

**qwen3.6 GGUF has the same defect**, and supplies the missing `ref_size`
rather than omitting it: `real`, `ref_size [1920, 1080]`, boxes in norm-1000.
That is worse to consume, because the declaration is *complete* and therefore
looks trustworthy. Its MLX sibling — same weights — declares `norm1000`
correctly. The strategy below is written for nemotron and applies unchanged to
qwen3.6 GGUF; the lesson is that this is a **class** of failure, not one model's
quirk, and that it can be introduced by the serving path.

Its actual output for the 1920×1080 fixture:

```
ANCHOR [65, 145, 225, 333]     DYNAMO [105, 567, 255, 792]
BEACON [325, 133, 475, 300]    EMBER  [385, 567, 535, 850]
CIPHER [600, 145, 795, 392]    FALCON [695, 567, 905, 812]
```

### The strategy: infer the frame, do not trust the declaration

Apply in order. Each step is cheap and the first that resolves wins.

**1. Aspect test — the discriminator.** Compare the aspect of the *observed
coordinate extent* against the aspect of the image.

- real (any frame) → extent aspect ≈ image aspect
- normalized (`norm1`/`norm1000`) → extent aspect ≈ **1.0**, because the
  coordinate space is square regardless of the image

Measured on nemotron: image aspect **1.778**, observed extent aspect **1.065**.
Normalized, decisively, despite declaring `real`.

Requires a non-square image with objects spread across the frame — which is
what `scene_hd` is, and why it anchors this probe. On a square image the test
cannot discriminate and you fall through to step 2.

**2. Range test.** Max coordinate ≤ 1.0 → `norm1`. Max ≤ 1000 while the image
exceeds 1000 in either axis → `norm1000`. Ambiguous when `max(W, H) ≤ 1000`,
where real and norm-1000 occupy the same range — do not use it alone.

**3. `implied_scale`.** If the boxes are the right *shape* in the wrong *frame*,
the uniform factor recovers them: divide by it and re-score. This is what turned
the qwen3.8 GGUF case from a 0.079 miss into a 0.909 hit before `ref_size`
existed. `bbox_contract` records `implied_scale` and `iou_at_implied_scale` on
every run, so a frame error is never again reported as a bare miss.

**4. Per-model calibration, cached.** Steps 1–3 are runtime heuristics; the
durable fix is to measure each model once against a known fixture, record the
dialect it actually emits, and apply that thereafter — re-validating whenever
the model, quantisation or payload changes, since the qwen3.8 GGUF result shows
the frame can be a property of the *serving path*, not only the weights.

### What not to do

Do not "fix" nemotron by relaxing the scorer to best-fit and moving on. Best-fit
is what hid this class of error in the first place: it silently rescues a model
that describes its own output incorrectly, and the caller — who cannot run a
best-fit search against ground truth they do not have — gets wrong boxes.
Record the mismatch, then correct it deliberately.

## Reading the probe's metrics

`bbox_contract`, its `bbox_contract_multi` reproducer and its
`bbox_contract_reasoning` control all emit:

| key | meaning |
| --- | --- |
| `declared_type` / `declared_order` / `declared_ref` | what the model claimed |
| `hits_declared` / `iou_declared` | grounding scored **only** in the declared dialect |
| `hits_bestfit` / `bestfit_dialect` | legacy search over type × order — the gap is the cost of a wrong declaration |
| `implied_scale` / `iou_at_implied_scale` | frame-error diagnostic (see step 3) |
| `declaration_matches_boxes` | the load-bearing one: declaration agrees with the numbers |
| `contract_followed` | matches, and loses nothing against best-fit |
| `field_name` | `box_2d` \| `bbox_2d` \| `bbox`, recorded rather than mandated |

A model that scores `hits_bestfit` 6/6 with `contract_followed` false has
**perfect vision and an unreliable self-description**. That is nemotron's row
and qwen3.6 GGUF's, and it is a different defect from a low `hits_bestfit`,
which is genuinely poor grounding.

The converse also occurs and is easy to misread: qwen3.8 GGUF scores
`hits_declared` 6/6 with `hits_bestfit` **1/6**. A low `hits_bestfit` is not
evidence of poor grounding when `ref_size` is present — it can simply mean the
frame is one no dialect search would try.

## Limits of what was measured

- One fixture, one image size, one prompt, think-off. `n=1` per configuration
  except `bbox_contract_multi`, which is `n=3`. Conventions could vary with
  image size or prompt shape — and gemma4's axis flip proves prompt shape alone
  is enough to change one.
- The failure **is** covered by a committed test, and it is corpus-wide. Under
  the `bbox_type`/`ref_size` schema, `bbox_contract_multi` — distractor images
  attached with an instruction to ignore them — is followed in only **5 of 21**
  runs across seven configurations
  ([campaign](vision-campaign-2026-08-16-seven-model.md)):

  | model | engine | followed | how it fails |
  | --- | --- | --- | --- |
  | gemma4:31b-it-q4_K_M | GGUF | 0/3 | declares `xyxy`, emits `yxyx` |
  | gemma4:31b-nvfp4 | MLX | 2/3 | `yxyx` on one run of three |
  | qwen3.6:35b-a3b-q4_K_M | GGUF | 0/3 | declares `real` [1920,1080], emits norm-1000 |
  | qwen3.6:35b-a3b-nvfp4 | MLX | 0/3 | declares `real` [1024, 768] — a frame never sent |
  | qwen3.8:27b-q4_K_M | GGUF | 0/3 | declares `norm1000`, emits real in a ~1.33× frame |
  | qwen3.8:27b-nvfp4 | MLX | 3/3 | — |
  | nemotron3:33b-q4_K_M | GGUF | 0/3 | declares `real`, no `ref_size`, emits norm-1000 |

  No GGUF configuration passes, in nine attempts. For qwen3.8 GGUF the
  single-image probe passes 3/3 and `bbox_contract_reasoning` — where the model
  must *use* the other images — passes 3/3, so neither image count nor reasoning
  load is the trigger on its own. What distinguishes the failing cell is being
  told to **ignore** the other images.

  The mechanism is unexplained; the rates are reproducible. This
  characterisation changed three times before repeats were run, each earlier
  version drawn from a single observation — the rates are the record, not the
  narrative. The gemma4 MLX cell is the standing reminder: the one-shot sweep
  caught its 1-in-3 outcome and would have been written up as "always flips".

- qwen3.8 GGUF's declared frame is itself approximate and unstable: 2500×1406,
  2560×1440 and 2324×1312 across runs on the same 1920×1080 input, all ≈1.30–1.33×.
  It knows it is working in a rescaled frame and does not know the frame exactly.
