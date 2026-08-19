# ADR 0030: bbox conformance is scoped to image geometry, and the norm-1000 pin is what makes it geometry-independent

- **Status:** accepted 2026-08-19. Extends
  [ADR 0027](0027-bbox-requests-pin-norm1000-and-carry-an-anchor.md), which
  decided the pin and the anchor but measured both at a single image size.
  Specified as SPEC clauses C13–C18 in
  [vision-bbox-response-contract.md](../spec/vision-bbox-response-contract.md)
  §4 and measured by `bbox_contract_anchored_1img` and
  `bbox_contract_real_1img` across 14 geometries. Campaign:
  [vision-campaign-2026-08-19-geometry.md](../vision-campaign-2026-08-19-geometry.md).
- **Date:** 2026-08-19

## Context

Every rate in ADR 0027 and in the contract SPEC — the 21/21, the 11 of 26, the
42/42, the 1 of 439, the 107 anchored cells — was measured on one fixture,
`scene_hd`, at 1920×1080. The SPEC said so in a closing footnote and left the
consequence unexamined.

That is not a hypothetical gap. The contract exists so a caller can convert
coordinates from an image **nobody has measured**, and in practice such images
arrive at whatever size they happen to be — pasted into a chat window, cropped
from a screenshot, shot on a phone. A conformance rate established at 16:9 HD
says nothing about that population, and 1920×1080 turns out to be an unusually
friendly point in it: it is not even patch-aligned (`1080/32 = 33.75`,
`1080/48 = 22.5`), so the corpus had never been measured on a clean grid either.

14 geometries were measured, in two tiers: controlled pairs that isolate one
variable each (an alignment twin per stride, a portrait with the same pixel
count as `hd`, sizes above and below the token budget), and six sizes drawn once
from `random.seed(20260818)` and frozen as literals to stand in for pasted
images. Both qwen families, both think modes, temperature 0.

## Decision

**1. A conformance rate is meaningless without its geometry, and both the
geometry and the token-budget configuration are recorded on every cell.**
`geometry`, `image_size`, `image_aspect` and `label_px_clamped` are written into
each score block, for the same reason [ADR 0012](0012-benchmark-report-templates.md)
rule 6 puts `num_ctx` there: a number whose meaning depends on an unrecorded
setting is not a measurement.

**2. Production bounding-box requests MUST pin norm-1000. This is now a
correctness requirement, not a declaration-honesty preference.** ADR 0027
justified the pin on declarations being usable 21/21 against 5/21 for free
choice. The geometry sweep gives a much stronger reason:

| pin | cells converting 6/6 |
|---|---|
| **norm-1000** (`bbox_contract_anchored_1img`) | **55 of 56** across 14 geometries, both models, both think modes |
| `real` pixels (`bbox_contract_real_1img`) | qwen3.8 **14/14**; qwen3.6 **1/14** |

Under the norm-1000 pin the model never has to name the frame it is working in,
so its internal resize cannot contaminate the coordinates and image geometry
stops being a variable at all. Under a `real` pin the model must state a frame,
and that is precisely where the resize leaks.

**3. `real` + `ref_size` is not a supported production path on this corpus.**
It is retained as a diagnostic arm because it is the only condition that exposes
the frame. It must not be recommended to callers.

**4. C7 (`bbox_self_check`) is a strong filter whose failure rate is
geometry-dependent, and it MUST NOT be quoted as a single number.** ADR 0027's
amendment recorded one silent failure in 107 anchored cells. Under the `real`
pin across geometry, qwen3.6 produced **four silent failures in fourteen
cells** — `self_check` passing while the anchor converted only 2–3 boxes of 6.
The mechanism is unchanged from the amendment (a fabricated frame whose aspect
happens to match the objects' extent defeats both checks); what geometry changes
is how often it happens. The 1-in-107 figure is a property of 16:9 HD, not of
the validator.

**5. The anchor's value is per-model and must be measured, never assumed.**
Across the same 14 geometries, under the `real` pin:

- **qwen3.8** — anchor converts 6/6 at every geometry while best-fit dialect
  search reaches only 1/6 in five of them. The anchor is doing essential work.
- **qwen3.6** — anchor converts 6/6 at one geometry; best-fit reaches 6/6 at
  **all fourteen**. Here the anchor is the worse path.

This does not overturn [C9](../spec/vision-bbox-response-contract.md) — best-fit
remains a diagnostic, because a caller with no ground truth cannot tell which of
these two models it is holding. It does mean "the anchor rescues the response"
is a qwen3.8 finding that was generalised too far.

## Consequences

- Geometry results are **not comparable** to the rates in ADR 0027: those are
  three-image distractor-condition cells, and the geometry arms send one image
  so that changing the scored image's size does not also change image count and
  the distractors' geometry. Comparing the two conditions at `hd` is what
  isolates the distractor effect, which no cell had previously done.
- C17 is settled and the SPEC's prior wording was wrong: frames **smaller** than
  the input are common, not absent. The reported frame spans BOTH directions:
  0.46×–2.50× on qwen3.6 and 0.67×–1.44× on qwen3.8, with 8 of 13 qwen3.6 cells
  and 6 of 13 qwen3.8 cells below 1.0×. The claim that every observed anchor reported a frame larger than the
  input was an artefact of only ever sending 1920×1080.
- Fixtures for the geometry family are generated by `gen_geometry.py` into
  `visimgs/geom/` and never into `visimgs/`. The two corpora were rendered with
  DIFFERENT fonts and that is recorded, not assumed: the original scored corpus
  (`visimgs/*.png`) was rendered against DejaVu on Linux, while this geometry
  corpus records `Arial.ttf` (sha256 `525979822591a344`) because it was rendered
  on macOS, where the DejaVu path `gen_scenes.py` hardcodes does not exist. That
  is also why the generator must never write into `visimgs/`: re-running
  `gen_scenes.py` on this host would silently replace every measured image with a
  different-font render. Geometry results are therefore not glyph-comparable with
  the scored corpus, which is harmless here — §4.3 already bars text-dependent
  metrics from gating conformance.
- Text-dependent metrics do not gate conformance at any geometry other than
  `hd`: glyph size scales with the fixture, and a 14 px serial is ~2 px at
  320×320. Label size is clamped at 11 px and the clamp is recorded per cell, so
  a shape task stays well-posed without a legibility failure being misread as a
  contract failure.

## Alternatives considered

**Vary geometry inside the existing three-image arms.** Rejected: it moves image
count, total token load and the distractors' own geometry at the same time, so
nothing in the resulting cell could be attributed to the scored image's shape.

**Keep one fixture and argue geometry-invariance from the encoder's resize
behaviour.** Rejected — and the measurement shows why. The reported frames are
not a smooth function of input size: qwen3.8 returns 2560×1440 for four
different inputs and 2337×1754 for two more, which no model of "resize by a
ratio" predicts.

**Test only small images.** Rejected. The two dead-band geometries found by the
random draw, `paste3` (1235×1181) and `paste4` (2750×2379), are both large. The
condition that disables C7's aspect check is near-squareness, not size.
