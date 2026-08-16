# ADR 0027: bounding-box requests pin norm-1000 and carry a self-calibrating anchor; `ref_size` is never trusted

- **Status:** accepted 2026-08-16, on fork `main`. Exercised by
  `bbox_contract_pinned`, `bbox_contract_perobject` and `bbox_contract_anchored`
  in `docs/maxusai/vision-suite/vision_suite.py`.
- **Date:** 2026-08-16
- **Deciders:** MaxusAI fork maintainers
- **Related:**
  [vision-bbox-coordinate-conventions.md](../vision-bbox-coordinate-conventions.md)
  (the consumer-side procedure this ADR fixes),
  [campaign 2026-08-16 seven-model](../vision-campaign-2026-08-16-seven-model.md)
  (the measurements this rests on),
  [ADR 0022](0022-thinking-is-off-for-vision-work.md) (think off for vision — all
  rates here are think-off),
  [ADR 0012](0012-benchmark-report-templates.md) (report templates)

## Context

Every vision model in this corpus locates shapes accurately and reports those
locations in a convention of its own choosing. Asked to declare that convention,
five of seven declare it correctly on a single image — and under the one
condition that reliably breaks them (distractor images attached with an
instruction to ignore them) only **5 of 21** runs declare it correctly.

The failures are not grounding failures. `hits_bestfit` is 6/6 in nearly every
failing cell: the model found all six shapes and described its own output
wrongly. Two failure directions matter, and they are not equally recoverable:

- **Declare `real`, emit normalized** (nemotron3, qwen3.6 GGUF). Silently halves
  every box. qwen3.6 GGUF supplies a complete-looking `ref_size [1920, 1080]`,
  which is *worse* to consume than nemotron's omission because it looks
  trustworthy.
- **Transpose the axes** (gemma4, both engines). Emits `yxyx` while declaring
  `xyxy`, conditionally — the single-image probe is honest and the distractor
  probe is not.

The axis case is the one that forced this decision. **No numeric heuristic
detects it.** A transposed box in a normalized space has the same range, the
same extent aspect and no scale error, so the aspect test, the range test and
`implied_scale` all see nothing. It is detectable only against ground truth,
which a caller does not have on an arbitrary image.

Measured 2026-08-16, seven configurations × 3 repeats, think-off, server
`0.32.5-maxusai-a5d65906`, powermode 2, cold restart per model:

| probe | declaration | `contract_followed` |
|---|---|---|
| `bbox_contract_multi` | free choice, top-level | **5/21** |
| `bbox_contract_pinned` | pinned norm-1000, top-level | **21/21** |
| `bbox_contract_perobject` | pinned norm-1000, per object | **21/21** |
| `bbox_contract_anchored` | pinned, named keys, `__IMAGE__` anchor | **21/21** |

Two results are worth stating plainly because both contradict what was expected:

**Declaration placement does nothing.** Top-level and per-object are 21/21 each,
with identical declared values. An earlier ad-hoc run appeared to show
per-object rescuing top-level 3/3 vs 0/3. It does not reproduce: that prompt
lacked the explicit statement of the space that both arms now share, and that
sentence — not the placement — was the active ingredient.

**Stating what norm-1000 *is* changes behaviour.** The wording that moved 5/21
to 21/21 is *"each axis scaled independently to 0-1000, x by 1000/width and y by
1000/height. The coordinate space is 1000x1000 whatever the image's shape is."*
Both axes are divided by their own dimension, so the space is square and does
**not** preserve aspect ratio.

## Decision

**Bounding-box requests pin the convention and carry their own calibration. A
consumer derives the coordinate space from the response; it never trusts a
declared `ref_size`.**

1. **Pin norm-1000 explicitly, and state what the space is.** Do not offer the
   model a choice of convention. The statement of the space is part of the
   contract, not commentary — it is what the 5/21 → 21/21 result rests on.
2. **Request named coordinates** — `x1`, `y1`, `x2`, `y2` as separate fields,
   never a positional array. A positional array is the sole reason `coord_order`
   has to exist; named fields make the transposition *unrepresentable* rather
   than discouraged.
3. **Require an `__IMAGE__` anchor** covering the whole image, in the same
   convention. It is the one box whose answer is known on any image, so it
   calibrates an image nobody has measured, for the cost of one box:
   `[0,0,~1,~1]` → `norm1`; `[0,0,~1000,~1000]` → `norm1000`; `[0,0,X,Y]` →
   `real` in a frame of X × Y.
4. **Derive, do not trust.** Convert using the space the anchor implies and
   ignore any declared `ref_size`. qwen3.8 GGUF reports that frame as
   2500×1406, 2560×1440 and 2324×1312 across runs on one 1920×1080 input;
   nemotron3 omits it while declaring `real`.
5. **Placement of the declaration is free.** Measured null; choose whichever
   suits the schema. Per object is marginally more robust to truncation, which
   is the only reason to prefer it.

## Consequences

- The heuristic ladder in the conventions doc becomes the **fallback**, used
  only when the anchor is missing or malformed. It was never sufficient — it
  cannot see an axis swap — and it is retained for responses that predate this
  contract.
- Per-model dialect calibration is no longer required for correctness. It stays
  useful as a regression signal.
- One extra object per request. That is the whole cost.
- Compliance is now checkable **per request** rather than per model: a response
  whose anchor disagrees with its declaration is rejectable without ground
  truth.
- **Not established: that the anchor rescues a real failure in the field.**
  `anchor_beats_declared` is false in all 21 cells, because under pinning
  nothing lied and the anchor never had to fire. Its recovery behaviour is
  verified only against synthetic responses — a payload declaring `norm1000`
  while emitting real pixels in a 1.302× frame recovers to 6/6 at IoU 0.997 with
  `ref [2500, 1406]` derived. What is measured is that all seven configurations
  *comply* with the protocol (21/21 named coordinates, 21/21 anchors at exactly
  `[0,0,1000,1000]`, nemotron3 included). Do not cite the anchor as a
  demonstrated field rescue.
- Rates are one fixture at one image size, think-off. A square image, or one
  where the anchor is ambiguous between `real` and `norm1000` because
  `max(W,H) ≤ 1000`, is not covered.

## Alternatives considered

- **Keep free choice and score best-fit.** Rejected: best-fit is what hid this
  class of error originally. It silently rescues a model that describes its own
  output incorrectly, and a caller without ground truth cannot run the search.
  It also fails outright where it is most needed — qwen3.8 GGUF's 1.3× frame
  scores `hits_bestfit` 1/6 while the declaration scores 6/6.
- **Mandate `ref_size` and trust it.** Rejected on evidence: the one model that
  reports a frame reports a different one each run, and the two models that get
  the type wrong also get `ref_size` wrong or absent.
- **Move the declaration per object instead of pinning.** Rejected: measured
  null, 21/21 either way.
- **Detect the convention from the numbers alone.** Rejected: sufficient for
  normalized-vs-real given a non-square image with spread objects, and blind to
  axis order in every case.
- **Pin `real` in the caller's own frame.** Rejected: Qwen-VL's documented
  behaviour is to answer in the *resized* frame, so `real` re-introduces exactly
  the frame ambiguity the anchor exists to remove.
