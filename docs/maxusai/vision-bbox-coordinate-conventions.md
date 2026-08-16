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

`bbox_contract`, think-off, single 1920×1080 image, six labelled shapes,
server `0.32.5-maxusai-1de352ef`, powermode 2.

| model | engine | declared | `ref_size` | hits | IoU | contract |
| --- | --- | --- | --- | --- | --- | --- |
| gemma4:31b-it-q4_K_M | GGUF | `norm1000/xyxy` | — | 6/6 | 0.961 | ✅ |
| gemma4:31b-nvfp4 | MLX | `norm1000/xyxy` | — | 6/6 | 0.962 | ✅ |
| qwen3.8:27b-q4_K_M | GGUF | `real/xyxy` | **[2500, 1406]** | 6/6 | 0.971 | ✅ |
| qwen3.8:27b-nvfp4 | MLX | `norm1/xyxy` | — | 6/6 | **0.990** | ✅ |
| qwen3.6:35b-a3b-nvfp4 | MLX | `norm1000/xyxy` | [1920, 1080] | 6/6 | 0.964 | ✅ |
| nemotron3:33b-q4_K_M | GGUF | `real/xyxy` | **missing** | **0/6** | — | ❌ |

Five of six are honest, across **four** conventions. Every one of the six
located all six shapes.

### qwen3.8 GGUF reports in a 1.302× frame, and will say so

It declares `ref_size [2500, 1406]` for a 1920×1080 input — 1.302 on both axes.
That is the same factor that made its `multi_3img` DYNAMO box look like a 250px
miss: raw IoU **0.079**, but **0.909** once divided out. The shape was always
found; only the frame was wrong.

This is why `ref_size` is mandatory for `real` rather than optional. "Absolute
pixels" is not a convention — it is a convention *plus a frame*, and Qwen-VL's
documented behaviour (coordinates relative to the resized image) means the frame
is routinely not the one the caller sent.

### gemma4 emits `xyxy`, not the documented `yxyx`

On both engines. Gemma's `box_2d` is documented as `[ymin, xmin, ymax, xmax]`,
but given an explicit choice and a duty to declare, gemma4 used `xyxy` and
declared it honestly. Treat `yxyx` as a **chat-template artifact, not a fixed
property of the model** — do not hard-code it.

## Recovering coordinates from nemotron3

nemotron is the one model that gets the declaration wrong, and it gets it wrong
in the dangerous direction: it declares `real`, omits the required `ref_size`,
and its boxes are actually **norm-1000**. `hits_bestfit` is 6/6 on
`norm1000/xyxy` — it **located every shape** and described the convention
wrongly. A consumer trusting the declaration would place every box at roughly
half scale.

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

`bbox_contract` (and its `bbox_contract_multi` control) emit:

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
**perfect vision and an unreliable self-description**. That is nemotron's row,
and it is a different defect from a low `hits_bestfit`, which is genuinely poor
grounding.

## Limits of what was measured

- One fixture, one image size, one prompt, think-off, `n=1` per configuration.
  Conventions could vary with image size or prompt shape.
- The single-image probe does **not** reproduce the failure seen under
  cross-image reasoning, where qwen3.8 MLX declared `absolute` while emitting
  normalized boxes. `bbox_contract_multi` attaches distractor images and both
  builds stay honest, so image count is not the trigger — the reasoning load is.
  That condition is bracketed but not yet covered by a committed test.
