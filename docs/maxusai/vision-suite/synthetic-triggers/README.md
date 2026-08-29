# Synthetic triggers for the qwen2.5-vl fp16-accumulate class

Reproducible, data-free images that drive the qwen2.5-vl vision tower's
fp16-accumulate GEMM overflow (`docs/maxusai/qwen25vl-3b-poison-image-garbage-decode.md`,
issue #216). Generators here; `results-*.jsonl` are the measured sweeps.

## Why these exist

The fp16 overflow depends on GEMM **summation order**, so each serving engine
(the removed Go vision engine ≤0.24, the llama-server/clip path ≥0.30) has its
own **disjoint** poison set. A trigger proves the *class*, not a universal
image. These first-order periodic patterns (checkerboards, stripes, rings, dots
at pitches 14–112, geometries from 616×1288 up to 3584×3584 = the 16,384-token
model max) were swept against every stock release with the model **fully
GPU-resident** (guarded: a run aborts if the mmproj offload is disabled, since
CPU vision is fp32 and would score every image trivially healthy).

## Matrix (stock `ollama/ollama`, `qwen2.5vl:3b-q4_K_M`, 54 candidates each)

| Release | Engine | Synthetic X / 54 | Fingerprint | Note |
|---|---|---|---|---|
| 0.7.1 | Go | 0 (4 waves, 159 candidates) | — | capped at ~1 MP, so the size axis is unavailable — see below |
| 0.24.0 | Go | 7 | `!`×30 | `checker_p14_1350x1800` reproduces it — falsifies "0.24 was clean" |
| 0.30.0 | llama-server (clip) | 9 | `?`×30 | the PR-body `checker_p56_1350x1800` fires here |
| 0.32.9 / 0.32.15 / 0.33.0 | llama-server (clip) | (prior) | `?`×31 | the `checker_p56_1350x1800` PR trigger + preflight `poison_probe` |

Two glyphs, one mechanism: the Go engine emits `!`×30 and the clip path `?`×31,
both the single-repeated-glyph degenerate decode with `done_reason: null`.

## 0.7.1 resisted four waves — and the reason is `max_pixels`, not immunity

| Wave | Candidate class | vs 0.7.1 |
|---|---|---|
| 1 (`gen.py`, `gen_large.py`) | periodic checker/stripe/ring/dot, pitch 14–112, up to 3584² | 0/54 |
| 2 (`gen_wave2.py`) | sub-patch pitches 1–7, off-axis rotations 15/30/45°, plaid, chirps, seeded noise, **color** clash/opponent, text-rows | 0/57 |
| 3 (`gen_wave3.py`) | natural statistics **at ≤1 MP native**: 1/f pink noise, specular highlights, voids, localized edge clusters, color blobs, grad+texture, and a "wood grain + thin cables" mimic of the actual corpus poison photo | 0/36 |
| 4 (`gen_wave4.py`) | the wave-3 set plus periodic controls, all at **2048×2048** | 0/12 |
| 5 (`gen_wave5.py`) | max-contrast periodics (pitch 7/14/28/56, checker + stripes, token-grid aligned and 14 px anti-aligned) at four **~1.0 MP** geometries, every side ÷28, so 0.7.1 serves them **native at full frequency** | 0/40 |

**Why no synthetic reaches 0.7.1 — its token budget is half the others'.**
Measured on the identical image (1350×1800):

| engine | tokens served | vs 0.7.1 | synthetically triggerable |
|---|---|---|---|
| **0.7.1** | **1265** (~0.99 MP) | — | **no** (0 / 199 candidates, 5 waves) |
| 0.24.0 | 2624 (~2.06 MP) | 2.1× | yes — `checker_p14`, `stripes_v_p56` |
| 0.33.0 | 3100 (~2.43 MP) | 2.45× | yes — `checker_p56`, `stripes_v_p56` |

Two compounding effects, both measured here: 0.7.1's reductions are **2.1–2.45×
shorter** (fewer accumulation steps to overflow in), and anything above ~1 MP
is **downscaled before the tower sees it**, averaging away the high-frequency
contrast that drives the peaks. Wave 5 removed the second effect entirely —
native, full-frequency, maximum-contrast, both grid alignments — and still
scored 0. So this is not a search failure: **0.7.1's budget sits below the
synthetic overflow threshold**, and its only known triggers are razor-thin
natural images (basin ≈ a point; see the perturbation study).

0.7.1 is therefore the natural experiment for the central claim: **the token
budget is the exposure dial.** Halve the budget and uniform synthetic patterns
stop reaching the cliff; raise it (2048², 2560², the 16,384-token model max)
and more images cross it.

**The measured explanation** (this is why `prompt_eval_count` is captured):
the Go engine caps every image at **~1,260–1,265 tokens ≈ 0.99 MP** — exactly
the stock `max_pixels` default of `1280 × 28 × 28 = 1,003,520`.

| source | tokens served | effective |
|---|---|---|
| 1288×616 (0.79 MP) | 1047 | native |
| 1350×1800 (2.43 MP) | 1265 | 2.5× downscale |
| 2048×2048 (4.19 MP) | 1260 | **4.2× downscale** |
| 2560×2560 (6.55 MP) | 1260 | 6.6× downscale |

Downscaling averages adjacent black/white cells to mid-gray, destroying the
high-frequency contrast that drives the activation peaks — so on this engine a
large synthetic is tested only in blurred form, and **the size axis is simply
unavailable**. Its known trigger (corpus `04431b0d…`, 1288×616 = 0.79 MP) sits
*under* the cap and arrives native.

**Upstream consequence, stated plainly: `max_pixels` is an exposure dial.** The
old engine's conservative ~1 MP cap is partly protective against this class;
raising the ceiling (2048², 2560², or the 16,384-token model maximum) both
lengthens the GEMM reductions and preserves the high-frequency content that
pushes the final block over the fp16 ceiling. That is the same axis wave 1
measured *positively* on the clip path, where `checker_p28` and `rings_p5` fire
only at ≥2048².

**Positive control — the negatives are real.** A sweep that cannot detect an X
would also report 0/159. So the same harness, same container, same grading was
run on 0.7.1 against both corpus triggers (`results-071-positive-control.jsonl`;
image content redacted, it is client data):

| image | size | 0.7.1 verdict |
|---|---|---|
| `02c9d7e1…` (the **clip path's** poison) | 1008×756, 1007 tokens | **H** — and described correctly |
| `04431b0d…` (0.7.1's own poison) | 1288×616, 0.79 MP native | **X**, `!`×31, `done_reason: null` |

That is the disjoint-set claim demonstrated in a single cell: each engine
serves the other's poison perfectly and fails its own. It also proves the
harness detects an X on this engine, so 0/159 is a true negative, not a blind
instrument.

**Geometry is not sufficient — content is the trigger.** All **29** corpus
images that share the poison's exact geometry (1288×616 = 0.79 MP, identical
token count, identical GEMM reduction lengths) were run against 0.7.1:
**1 X / 29** — only `04431b0d…` itself
(`results-071-geometry-cohort.jsonl`, descriptions redacted). So the size axis
is a *risk multiplier* (longer reductions, more chances to overflow — measured
positively on the clip path) but within a fixed size only specific pixel
content resonates with the final block's outlier channels. Consistent with
#214's original observation that a pixel-identical lossless re-save still
triggers, i.e. the trigger is pixel content, not container or shape. It also
puts a rough floor under the base rate for this engine at this geometry: ~1 in
29 corpus photos.

**The corpus trigger is decode-fragile — ship original bytes, not re-saves.**
Asked whether something was *wrong* with `04431b0d…` (bad encoding, metadata,
HEIC origin), we inspected and controlled it. The file is clean: baseline JPEG,
JFIF only, **no EXIF, no ICC, no orientation tag**, not progressive, valid EOI,
decodes without warnings. But the encoding control
(`results-071-encoding-control.jsonl`) is decisive:

| variant vs 0.7.1 | verdict |
|---|---|
| original bytes, verbatim | **X** (`!`×31) |
| lossless PNG re-save built from *libjpeg's* decode | H |
| JPEG re-encode q100 (max diff 4/255) | H |
| **lossless PNG built from *Go's own* decode** | **X** |

The reason is not the container — it is *whose* decode the re-save captured.
**Go's stdlib `image/jpeg` and libjpeg decode the same bytes to different
pixels**: 68.27% of pixels differ, max channel delta 18, mean 0.38 (IDCT and
chroma-upsampling rounding). Decoding the original with Go and re-saving that
as PNG gives a file measuring **0.00% difference, max 0** against Go's own JPEG
decode — and it triggers, exactly like the original
(`results-071-decoder-control.jsonl`).

So the causality is proven in both directions: same pixels → same failure in a
different container; ±0.38 mean LSB of decoder rounding → no failure at all.
Container and encoding are irrelevant; **pixel content is causal**, and the
margin is thin enough that a decoder swap crosses it. That fits `04431b0d…`
measuring 0.77× of the fp16 ceiling — the weakest known trigger — against the
synthetic checkerboard's 1.06×.

**Why the two decoders differ — a documented 256/255 luma gain, not a bug.**
The difference is not random rounding. Fitting `go = a·pil + b` over all 2.38 M
channel values gives **a = 1.003776**, and the differences are asymmetric
(+1 on 30.4% of values, −1 on only 2.1%) with the bias flipping by intensity
(dark −0.38, bright +0.39). Go's `image/color.YCbCrToRGB` explains it exactly:
instead of a constant ½ rounding adjustment it uses a value-dependent `257·Y′`,

    YY1 = 65536·Y′ + 257·Y′ = Y′ × 0x10101   →   gain 65793/65536 = 256/255 = 1.0039216

so that `Y′=0xff` maps to full-scale `0xffff` rather than `0xff80` (see the
comment block in Go's `ycbcr.go`). libjpeg rounds conventionally. Both are
legal readings of JFIF; the measured 1.003776 falls just under the theoretical
1.003922 because chroma terms and clamping dilute it. On top of that sits
chroma-upsampling disagreement at edges (this JPEG is 4:2:0; large deltas
concentrate where the local gradient is 33.1 versus 3.1 elsewhere, worst on B,
max 18).

**The gain is NOT the causal factor — tested and falsified**
(`results-071-gain-and-white-control.jsonl`). Scaling libjpeg's healthy decode
UP by 256/255 does not make it trigger (H); scaling Go's triggering decode DOWN
by 255/256 does not stop it (X). Each variant behaves like its *source decode*,
not its brightness, so what flips the outcome is the fine per-pixel structure
(chroma-upsampling and IDCT residuals), not global luma. Nor is near-whiteness
alone sufficient: flat fields at 230/235/240/245/250/**255**, a 230→255
gradient, and a near-white field carrying the poison's own vent-louvre and
downlight structure are **all H** at 1288×616. The gain remains the correct
explanation for *why the decodes differ*; it is not the explanation for *which
images trigger*.

Three practical consequences: (1) report a corpus trigger as **original
bytes**, since a re-save preserves it only if it captured the pixels *that
engine's* decoder produces; (2) a "pixel-identical" claim must name the
decoder, or it is not a claim; (3) this is precisely why the synthetic matters
— a PNG of exactly specified integer pixels has no decoder ambiguity and
reproduces everywhere.

**How fragile? One LSB, or a mirror.** Perturbing Go's decode of the poison
(as PNG, so no re-encode confound) destroys the trigger every way we tried
(`results-071-perturbation.jsonl`):

| perturbation | what it preserves | 0.7.1 |
|---|---|---|
| none (control) | — | **X** |
| random noise **±1 LSB** | everything else | H |
| noise ±2, ±4 | everything else | H |
| **horizontal flip** | the *exact* histogram and every local structure, mirrored | **H** |
| vertical flip | same | H |
| shuffle 28×28 patches | each patch's content (token grid positions randomized) | H |
| shuffle within patches | patch positions | H |
| global pixel shuffle | the exact histogram only | H |

The flips are the informative ones: a mirror changes no pixel *value* and no
local neighbourhood, yet it heals the image — so the trigger is not a property
of the value distribution, the patch contents, or the texture statistics. It
depends on the **exact pixel arrangement** reaching specific positions, i.e. on
where content lands in the token grid and how the reduction accumulates across
it. Combined with ±1 LSB sufficing to heal, the poison basin in pixel space is
essentially a point, not a region.

That fully explains the 159 negative synthetics: no statistical family — pitch,
rotation, spectrum, noise, colour, natural statistics — can be *aimed* at a
basin this thin. It also settles the shipping question. A corpus trigger is
useless to a bug report (one re-save, one resize, one decoder, one flip and it
is gone); the synthetic checkerboard at 1.06× survives every decoder, container,
engine ≥0.30 and both GPU generations.

**What is NOT concluded:** that 0.7.1 is safe. It fails its own corpus image at
0.79 MP, and four waves of first-order, sub-patch, rotated, mixed-spectrum,
noise, color, and natural-statistics synthetics simply did not land inside its
(smaller, cap-bounded) poison set.

## Findings

1. **Image size is causal.** Several patterns garble *only* at large geometries:
   `checker_p28` and `rings_p5` need ≥2048², `stripes_v_p14` (0.24) only at
   3584². Larger images → longer GEMM reductions → partial sums overflow even
   when stored maxima stay under the fp16 ceiling. Consistent with the
   activation measurements (a 1008→1800 px upscale of one image raised its
   final-block peak ~1.6×). Practical consequence: `max_pixels`-hungry OCR
   pipelines (1800–2560 px) sit in the worst part of the space.
2. **Some triggers cross engines.** `checker_p56` at 1350×1800 garbles the clip
   path, and the adjacent `checker_p14` garbles 0.24 — strong patterns overflow
   regardless of summation order.
3. **0/54 is not "clean."** 0.7.1's zero means *this pattern class* misses *its*
   set, not that the engine is safe — it fails its own corpus members. No engine
   is clean; the triggers move.

## Cross-engine trigger matrix — four engines, four different trigger sets

Every known trigger run against every engine generation, same harness, model
GPU-resident (`results-crossengine-*.jsonl`). Corpus triggers `39823be1`,
`74cee2fe`, `d55ff004`, `ead2a6c7` were isolated by the release-fold session.

| image | 0.7.1 (Go) | 0.24.0 (Go) | 0.30.0 (clip) | 0.33.0 (clip) |
|---|---|---|---|---|
| corpus `04431b0d` | **X** | H | H | H |
| corpus `39823be1` | H | **X** | H | H |
| corpus `74cee2fe` | H | **X** | H | H |
| corpus `d55ff004` | H | **X** | H | H |
| corpus `ead2a6c7` | H | **X** | **X** | **X** |
| corpus `02c9d7e1` | H | H | **X** | **X** |
| synthetic `checker_p14` | H | **X** | H | H |
| synthetic `checker_p56` | H | H | **X** | **X** |
| synthetic **`stripes_v_p56`** | H | **X** | **X** | **X** |

Four readings:

1. **The two Go releases are disjoint from each other**, not merely from the
   clip path — 0.7.1 and 0.24.0 share the same engine and share *no* trigger.
   ggml kernel drift between them is enough to re-roll which images fall over.
2. **0.30.0 and 0.33.0 are identical** across all nine images: one clip path,
   one trigger set, unchanged across three releases.
3. **Triggers vary in breadth.** `ead2a6c7` and `stripes_v_p56` cross the
   engine boundary; most triggers hit exactly one generation. Breadth tracks
   margin — the further above the fp16 ceiling, the more graphs it defeats.
4. **No engine is clean, and no engine is safe by generation.** Every column
   has at least one X.

## The shippable trigger (use this in bug reports)

`trigger_checker56_1350x1800.png` — committed here, 12 KB, generated by the
three lines below and **verified as the exact committed file** against stock
`ollama/ollama:0.33.0`: `'?'×31`, `done_reason: null`
(`results-033-shipped-artifact.jsonl`).

    md5    afc8ff7e84ee8958878b44675565d5b0
    sha256 d99b6b45e70b7cac2ee8771712f4caa6…
    1350×1800, PNG, black/white, 56 px pitch

```python
import numpy as np
from PIL import Image
ys, xs = np.mgrid[0:1800, 0:1350]
cells = ((xs // 56) + (ys // 56)) % 2
Image.fromarray(np.where(cells[..., None], 255, 0).astype(np.uint8).repeat(3, axis=2)).save("trigger.png")
```

**Widest coverage — `trigger_stripes56_1350x1800.png`** (11 KB, md5
`d8ee0feb7e23beceba74a55c641648d0`): vertical stripes, 56 px pitch, 1350×1800.
Verified **X on 0.24.0, 0.30.0 and 0.33.0** — the only known artifact, corpus
or synthetic, that spans the Go engine and the clip path. Use this one when the
claim is about the *class*; use the checkerboard when the target is current
ollama.

```python
import numpy as np
from PIL import Image
ys, xs = np.mgrid[0:1800, 0:1350]
Image.fromarray(np.where(((xs // 56) % 2 == 1)[..., None], 255, 0).astype(np.uint8).repeat(3, axis=2)).save("trigger_stripes.png")
```

Being PNGs of exactly specified integer pixels, they carry **no decoder
ambiguity** — unlike the corpus JPEGs, whose trigger depends on which decoder
reads them (see the decoder control above). Ship the file, or the three lines;
either reproduces bit-identically.

## Reproduce

```bash
python3 gen.py          # 40 base candidates (616×1288 … 1800×860)
python3 gen_large.py    # 15 large candidates (2048² … 3584²)
```

Then sweep a release with the model GPU-resident, grading each response as X
when `done_reason` is null or the content is one glyph repeated (the
`is_degenerate_decode` rule shared with the preflight `poison_probe`). A healthy
follow-up after each X confirms whether the slot is sticky (clip path
0.32.10–0.32.15) or recovers.

## Caveat / next

Each engine applies its own `max_pixels` smart-resize, so a large candidate may
be downscaled before the tower sees it — always read `prompt_eval_count` (every
`results-*.jsonl` carries it) before interpreting a size result. That check is
what explained 0.7.1's four negative waves.

Open: a synthetic member of 0.7.1's ≤1 MP poison set. Waves 2–4 covered the
obvious pattern families; the remaining leads are derivative — perturb the known
corpus trigger (crop, rescale, contrast/channel transforms) to map how far its
basin extends, or drive candidate selection with the activation harness
(`docs/maxusai/measure_qwen25vl3b_vision_activations.py`) instead of binary
probing, selecting for measured final-block peaks rather than guessing patterns.

## Blast radius: other clip-based vision models

Two questions the fix raises — is the class broader than qwen2.5-vl, and does
a shared-`build_ffn` patch help or perturb other models? Screened statically
with `fp16_audit.py` (GGUF only, no GPU), then probed with the checkerboard.

| model | F16 vision matmuls | unsafe at 50,688 | lowest threshold | checkerboard |
|---|---|---|---|---|
| **qwen2.5vl:3b** (known bad) | 194 | 2 | 60,757 `v.blk.17.ffn_gate` | **X** |
| qwen2.5vl:7b (healthy) | 194 | 2 | 61,651 `v.blk.17.ffn_gate` | H |
| **nemotron3:33b** | 195 | **20** | 54,445 `v.blk.16.ffn_down` | H (3,256 tok) |
| qwen3.6:35b-a3b | 165 | 2 | 52,733 `v.blk.2.mlp.linear_fc2` | H (2,368 tok) |
| gemma4:12b | **0** | 0 | — | H |
| granite3.2-vision | **0** | 0 | — | H |

Four readings:

1. **gemma4 and granite are structurally immune to this class** — their
   vision weights carry *no* F16 matmuls at all, so there is no
   fp16-accumulate GEMM to overflow. Whatever else they may do, it is not
   this.
2. **nemotron3 is the most exposed by weights** — 20 unsafe tensors, ten
   times qwen2.5-vl's two, with a threshold as low as 54,445. It did not
   garble on our trigger even at a 3,256-token budget, but that is the same
   position 0.24.0 was in before its own trigger was found: *untriggered is
   not immune*.
3. **The audit is a screen, not a predictor** — qwen2.5vl **7b** has an
   audit profile identical to 3b (194 matmuls, 2 unsafe) and is healthy in
   practice. Exposure needs weights *and* activations that land on the
   exposed channels; the 7B's peaks are lower.
4. **Scoping the ollama-side gate to `qwen25vl` is defensible today** — no
   other model is triggered by any known image — while patch 905's placement
   in the shared `build_ffn` is the right shape for the llama.cpp side,
   since it closes the same door for nemotron3 and qwen3.6 before anyone
   finds their triggers.

Caveat: nemotron3 and qwen3.6 are thinking models and spent the 20-token
budget on reasoning, so their answer *content* was not verified — only that
the degenerate `'?'`/`'!'` fingerprint was absent, which is the signal under
test.

### Margin test: every corpus trigger × exposed models, at stock and under forced fp16

The screen above used the synthetic only, at stock precision. Re-run with all
six corpus triggers plus the synthetic, at full image budget, twice: once at
stock (`GGML_CUDA_CUBLAS_COMPUTE_TYPE=auto` — the launcher gate stands down,
per-op `GGML_PREC_F32` protections intact) and once under maximum pressure
(`=f16` — forces fp16 compute on *everything*, stripping even upstream's
deliberate flash-attn `PREC_F32`).

| model | tokens served | failures at stock | failures under forced fp16 |
|---|---|---|---|
| nemotron3:33b (20 unsafe weights) | 3,259–3,305 | **0 / 7** | **0 / 7** |
| qwen2.5vl:7b | 1,064–3,100 | 0 / 7 | **0 / 7** |
| qwen3.6:35b-a3b | 1,055–2,371 | 0 / 7 | not run |
| **qwen2.5vl:3b** | 1,047–3,100 | **3 / 7** | **7 / 7** |

Three conclusions:

1. **The 3B has essentially no margin.** Under forced fp16 *every* image
   fails — including the four corpus photos it serves correctly at stock. It
   is not that a few images are unlucky; the model sits on the cliff and
   upstream's existing protections are all that keep most images on the safe
   side.
2. **Those existing protections do real work.** Stripping them (flash-attn
   `PREC_F32` among them) widens the 3B's trigger set from 3/7 to 7/7. That is
   an argument *for* the per-op approach of patch 905, not against it.
3. **nemotron3 and qwen2.5vl:7b have genuine margin, not luck.** Both survive
   maximum fp16 pressure on every known trigger — nemotron at 3.3 k tokens
   with ten times the exposed weights. So the ollama-side gate scoped to
   `qwen25vl` is not merely untested-elsewhere: the nearest candidates are
   measurably safe even when pushed harder than any real configuration.

Control for the whole set: the same harness, same image, same `=auto`, run
against qwen2.5vl:3b reproduces `X` on both known triggers with the runner
env showing `GGML_CUDA_CUBLAS_COMPUTE_TYPE=auto` — so these negatives are
measured without the fix active, not with it silently healing them.

### 3B-class sweep — typhoon-ocr1.5-3b is affected, and the gate fixes it

Six 3B-class vision models, all six corpus triggers plus the synthetic, at
stock precision (`=auto`, gate standing down):

| model | 02c9d7e1 | 04431b0d | 39823be1 | 74cee2fe | d55ff004 | ead2a6c7 | checker |
|---|---|---|---|---|---|---|---|
| **`scb10x/typhoon-ocr1.5-3b`** | H | H | H | H | H | **X** | H\* |
| `qwen2.5vl:3b-q4_K_M` | X | X | X | X | X | X | X |
| `qwen3-vl:2b-thinking` | H | H | H | H | H | H | H |
| `qwen3-vl:4b-thinking` | H | H | H | H | H | H | H |
| `qwen3.5:2b` | H | H | H | H | H | H | H |
| `qwen3.5:4b` | H | H | H | H | H | H | H |

**This overturns the earlier "typhoon does not reproduce on this estate"
conclusion** (PR #214 comment 5421905441). That sweep used photos and the two
qwen corpus triggers; it never tried `ead2a6c7` or the synthetic checkerboard,
both of which garble it. `typhoon-ocr1.5-3b` is the model reported in upstream
ollama#17687 — which therefore now has a reproducer requiring no private data.

**The shipped gate covers it.** ollama resolves that model to arch `qwen25vl`
(`/api/show` → `general.architecture: qwen25vl`), so `applyArchServerEnvs`
fires: with the gate active the runner env carries
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` and both failing images decode correctly.
Caution for anyone reading GGUF metadata directly: its *clip* metadata reports
`qwen2vl` after mmproj translation, which is not the string the gate keys on.

Its trigger set is narrower than the base model's (2/7 versus 7/7) —
consistent with a fine-tune shifting activation scales rather than removing
the fragility, and explaining why #17687's reporter hit it while earlier spot
checks here did not. The four newer small vision models are clean throughout,
supporting the `qwen25vl`-only scoping.


### Correction: the checkerboard does NOT trigger typhoon (\*)

The sweep above recorded typhoon failing on both `ead2a6c7` **and** the
checkerboard. On repeat, only the first holds. Five alternating repeats in a
fresh container:

| image | typhoon-ocr1.5-3b |
|---|---|
| corpus photo `ead2a6c7` | **X X X X X** — deterministic |
| synthetic checkerboard | **H H H H H** |

Three checks confirm the retraction rather than the original: a fresh container
with the checkerboard alone is healthy with and without the tracer; the slot is
**not** left poisoned by the preceding trigger (`checker → ead2a6c7 (X) →
checker` gives H, X, H); and six independent healthy runs followed the single
anomalous X. The sweep's second X was an artefact of one observation, not a
property of the image.

**Same node, narrower margin.** With the non-finite eval callback, typhoon's
failure on `ead2a6c7` lands on exactly the site the base model uses —
`ffn_down-31` (MUL_MAT, src0 `v.blk.31.ffn_down.weight` f16) — with **1 inf of
5,406,720** versus qwen2.5vl:3b's **10 inf** on the same image. Same mechanism,
less headroom, consistent with a fine-tune shifting activation scales.

**Methodological lesson, recorded because it cost a wrong public claim:** at one
overflowing element in 5.4 M the outcome is not reliably reproducible between
runs — plausibly cuBLAS kernel/split-k selection, which is not deterministic.
**Marginal cells need repeats before they are reported.** The upstream claim
that #17687 had gained a data-free reproducer was posted on a single
observation and had to be withdrawn
([ollama/ollama#18070 comment 5439413745](https://github.com/ollama/ollama/pull/18070#issuecomment-5439413745)).

What stands: typhoon-ocr1.5-3b **is** affected (deterministically, on a corpus
photo), is arch `qwen25vl`, and the shipped gate heals it. What does not:
#17687 still lacks a data-free reproducer — trigger sets are per-checkpoint,
just as they are per-engine.

### Do the two typhoon candidates share anything measurable?

Asked whether `ead2a6c7` and the checkerboard have a common property. They do
not — and once the checkerboard is retracted the question partly dissolves,
but the feature table is worth keeping because it shows how little global
statistics explain:

| image | triggers typhoon | mean | % extreme px | % flat | mean grad |
|---|---|---|---|---|---|
| checkerboard | no (retracted) | 127 | **100.0** | 96.5 | 9.1 |
| `ead2a6c7` | **yes** | 179 | **0.1** | 59.8 | 4.5 |
| `04431b0d` | no | 198 | 0.3 | 77.6 | 3.3 |

The two candidates sat at opposite extremes on every contrast measure, while
`04431b0d` — statistically the nearest neighbour of `ead2a6c7` — leaves typhoon
healthy. Consistent with the perturbation study: the trigger is not a
distributional property of the image but depends on which channels the
activations land in.

### Hardening the negatives: n=5 under forced fp16 (the retraction's other half)

The retraction above showed a single observation can read X when the truth is
H. The same logic applies in reverse: every "clean" verdict in the sweeps was
**one observation per cell**, so a model failing at, say, 1-in-5 would look
clean most of the time. Since the shipped gate deliberately *excludes* those
models, that is the claim most worth hardening.

Re-run at n=5, under `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16` — forcing fp16
compute on everything, harsher than any real configuration — on the three most
aggressive images (`repeat_sweep.sh`):

| model (all excluded by the gate) | checkerboard | `ead2a6c7` | `02c9d7e1` |
|---|---|---|---|
| nemotron3:33b (20 unsafe weights, 3.3 k tok) | HHHHH | HHHHH | HHHHH |
| qwen3.6:35b-a3b | HHHHH | HHHHH | HHHHH |
| qwen2.5vl:7b | HHHHH | HHHHH | HHHHH |
| qwen3-vl:2b-thinking | HHHHH | HHHHH | HHHHH |
| qwen3-vl:4b-thinking | HHHHH | HHHHH | HHHHH |
| qwen3.5:2b | HHHHH | HHHHH | HHHHH |
| qwen3.5:4b | HHHHH | HHHHH | HHHHH |
| **positive control** — qwen2.5vl:3b | **XXXXX** | **XXXXX** | — |

**105 consecutive healthy observations** across the seven excluded models, with
a positive control in the same container and the same forced-fp16 setting
firing 10/10. So the negatives are not "not observed once" — they are not
observed in five tries each, under conditions harsher than any deployment,
with the detector demonstrably working.

That is the standard the `qwen25vl`-only scoping now rests on.


## Running these probes yourself

Nothing here hardcodes a host path, a private corpus location, or a client
name. Two environment variables drive the scripts that need real inputs:

| variable | required by | meaning |
|---|---|---|
| `CORPUS_IMAGE_DIR` | `measure_channel_thresholds.py`, `simulate_fp16_accum.py` | directory of probe photographs. **No default** — the corpus these were developed against is private and deliberately not committed; point it at your own images. The scripts exit with a message rather than silently probing nothing. |
| `PROBE_DIR` | `measure_channel_thresholds.py`, `repeat_sweep.sh` | scratch directory holding derived inputs (`blk31_weights.npz`, decoded PNGs) |
| `QWEN_VL_MODEL` | the HF-based scripts | model id or local path; defaults to `Qwen/Qwen2.5-VL-3B-Instruct` |

The two committed trigger PNGs need none of this — they are generated
patterns and reproduce bit-identically from the snippets above.

**Data-handling rule for this directory:** a model's *description* of a client
image is client data, exactly as the image is. Result files here therefore
carry verdicts, token counts and degenerate-glyph fingerprints only; no
descriptive text, no client imagery, no client names, no host paths.

### Searching for a public trigger for `typhoon-ocr1.5-3b` — negative

`typhoon-ocr1.5-3b` (upstream ollama#17687's model) is affected, but its only
known trigger is a private customer photograph, so that issue still lacks a
data-free reproducer. Two targeted families were generated and probed
(`gen_typhoon_candidates.py`), n=3 per candidate at stock precision:

- **Structural mimicry of the known trigger** — cream wall, white ceiling
  band, saturated red/magenta mats, gold frames, documents behind glossy
  glazing, at three geometries including the trigger's own 1176×644.
- **Document pages**, since the model is OCR-tuned and #17687's reporter ran
  an 1800 px document pipeline — dense text at two page sizes, three point
  sizes and two paper tones, plus a document-with-colour-block variant.

**Result: 17 candidates × 3 = 51 probes, all H.** Controls in the same run
fired as expected: the corpus trigger gave `XXX` on typhoon, and the
checkerboard gave `XXX` on qwen2.5vl:3b, so the detector was live throughout.

This is what the perturbation study predicts. The poison basin is effectively
a **point** in pixel space — ±1 LSB or a mirror heals the base model's
trigger — so reproducing a trigger's *appearance* does not reproduce its
activation pattern. Mimicry is the wrong tool; only the exact pixels work.

**Practical consequence:** a public reproducer for typhoon would need either
activation-guided search against that checkpoint's weights (they are on
Hugging Face; the search would optimise pixels for overflow rather than
resemblance) or large-scale sampling of public document/photo corpora. The
hit rate for that sampling is **unmeasured** — the honest summary of what we
have observed on this model is 1 trigger among the handful of corpus images
probed, 0 across 51 synthetic probes, and 0 on the checkerboard over 5
repeats — so the sample is far too small to estimate a rate from, in either
direction. Neither route is needed to validate the fix: typhoon is arch
`qwen25vl`, the gate covers it, and healing was verified on the corpus image.

### Augmentation search, and why activation screening does not predict triggers

Attempt to turn trigger-hunting from a lottery into a recipe: augment the known
corpus trigger, watch how close `blk31.ffn_down` gets to the fp16 cliff, then
apply whichever augmentations help to a *public* base image.

**The metric.** Not the H/X verdict but the worst-case partial sum

    L1[t,r] = sum_c |x[t,c]| * |W[r,c]|     as a multiple of 65504

over the input `x` to `v.blk.31.ffn_down` and its f16 weight `W`. Below 1.0 no
accumulation order can overflow, so it reads distance-to-edge continuously
instead of only registering hits. Measured by hooking `blocks[31].mlp.down_proj`
in a Hugging Face bf16 forward (`augment_probe.py`), with augmentation and
patchify in DataLoader workers so the GPU stays fed.

**On that proxy the ranking is clean** (75 augmentations of the corpus trigger,
which sits at 0.870):

| augmentation | l1x | vs base | at-risk elems |
|---|---|---|---|
| `contrast_a2.00` | 1.236 | 1.42x | 2 |
| `scale_k2.00` | 1.229 | 1.41x | 19 |
| `contrast_a1.50 + bright+25` | 1.164 | 1.34x | 2 |
| `solarize_t224` | 1.070 | 1.23x | 1 |
| `contrast_a1.75` | 1.020 | 1.17x | 1 |
| `noise_sig1` | 0.873 | 1.00x | 0 |

Contrast peaks near alpha 2.0 and *falls back* by 2.5 as clipping binarises the
image; scale raises the count of at-risk elements far more than the peak (4x the
tokens = 4x the independent dot products, since the reduction length is fixed at
3420 and only the column count grows); noise does nothing, consistent with +-1
LSB *healing* the trigger in the perturbation study.

**But the proxy is falsified as a trigger predictor.** Loading typhoon's own
served vision tower out of the ollama GGUF into the HF module
(all 390 visual params mapped from the GGUF, `strict=True`) and
calibrating against ground truth:

| image | engine verdict on typhoon | l1x on typhoon's tower |
|---|---|---|
| corpus trigger | **X, 5/5** | **0.602** (`n_over` = 0) |
| generated checkerboard | H, 5/5 | **0.975** |

A confirmed trigger scores *lower* than a confirmed non-trigger, and the
worst-case bound calls the trigger provably safe. 177 of 342 public
combinations score above the real trigger; the seven highest were built and run
on the engine at n=5 and every one came back `HHHHH`. So the ranking above
describes the HF bf16 tower, not the served f16 one — plausibly because ollama's
preprocessing and f16 trajectory diverge from HF's over 32 blocks. **Do not use
HF activation screening to predict engine overflow.** Faithful screening has to
read the activations inside the engine, via the node meter now carried as
`llama/compat/801-clip-node-stats-meter.patch`.

### Slot poisoning is real and image-dependent (engine-measured)

While validating the above, the negative control fired. Running the documented
benign -> trigger -> benign order on a fresh container (typhoon, stock
precision, n=5 per cell) explains it:

| step | cell | n=5 |
|---|---|---|
| 1 | plain cert, **virgin slot** | `HHHHH` |
| 2 | gain candidate, virgin | `HHHHH` |
| 3 | corpus trigger (poison) | `XXXXX` |
| 4 | plain cert, post-trigger | `XXXXX` |
| 5 | gain candidate, post-trigger | `HHHHH` |
| 6 | plain cert, post-post | `XXXXX` |

The plain certificate is healthy alone and degenerate only after the trigger,
while the larger gain candidate stays healthy in the same poisoned slot — so the
carried state is not a blanket latch, and a sequential sweep can attribute one
image's failure to the next image. **Any sweep that reuses a slot needs this
ordering control**, or it will manufacture both false positives and, where a
recovery happens to land, false negatives.

## FOUND: a public trigger for `typhoon-ocr1.5-3b` (upstream ollama#17687)

`trigger_typhoon_c70_halfphase_1350x1800.png` — a 70 px black/white
checkerboard at 1350x1800, phase-shifted by **half a check (35 px) on both
axes**. Generated, deterministic, no fonts or external assets; regenerate it
byte-identical with `gen_typhoon_trigger.py` (md5
`0c95f8db6c3b0a1026c8faef0cef4a19`).

The half-phase shift is the whole trick. The identical board at phase 0
(`control_typhoon_c70_phase0_1350x1800.png`) is **healthy** — it makes a
paired positive/negative control that differs only by a 35 px translation.

### Evidence — 25/25, five independent virgin containers

Every run starts with a benign image in a fresh container to prove the slot is
clean, and any positive control runs **last** so it cannot contaminate the
candidate.

| build | precision | benign first | trigger, n=5 |
|---|---|---|---|
| `ollama/ollama:0.33.0` (stock) | default, no env | `HHH` | **`XXXXX`** |
| `ollama/ollama:0.32.15` (stock) | default, no env | `HHH` | **`XXXXX`** |
| fork `sync-0.33.0`, gate defeated | `=auto` | `HHH` | **`XXXXX`** |
| fork `sync-0.33.0`, gate defeated, different prompt | `=auto` | `HHH` | **`XXXXX`** |
| **fork `sync-0.33.0`, gate active** | f32 injected | `HHH` | **`HHHHH`** |

Reproduces on unmodified upstream with no environment variables, is
prompt-independent, and is healed by the `qwen25vl` f32 gate. It does **not**
fire `qwen2.5vl:3b` — the mirror image of the older checkerboard, which fires
the base model but not typhoon. Each finetune has its own trigger set.

### How it was found — measure inside the engine, not in HF

The earlier HF-proxy screen failed because it measured the wrong tower. The
fix was to extend the eval callback into a continuous per-node meter
(now shipped as `llama/compat/801-clip-node-stats-meter.patch`, env
`OLLAMA_CLIP_NODE_STATS=<name substring>`) and
run it under `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`, so nothing overflows and the
true magnitude at `ffn_down-31` is readable. Headroom is `max_abs / 65504`.
`libmtmd.so` is backend-agnostic, so this is a CPU-only build bind-mounted
over the image's own copy — no CUDA build.

Calibration was immediate and decisive, where the HF proxy had been inverted:

| image | engine verdict | in-engine headroom |
|---|---|---|
| corpus trigger (private) | X 5/5 | **1.201** |
| generated checkerboard c56 | H 5/5 | 0.749 |
| plain certificate | H 5/5 | 0.721 |

Then ~90 measurements at **3.9 s each** over check size, geometry, phase,
colour and photometry (`results-engine-meter.txt`). Check size is spiky, not
monotonic — c70 peaks at 0.919 while c56 sits at 0.749 — and the half-phase
offset supplied the last 9%:

| variant | headroom | engine |
|---|---|---|
| `c70` phase 0 | 0.919 | H 5/5 |
| **`c70` half-phase (35,35)** | **1.006** | **X 5/5** |

The meter's ordering predicted the engine exactly, including that phase 0
would *not* fire. Contrast and scale — the axes that dominated the HF
screen — move this metric only 0.721 → 0.731, which is why the earlier gain
experiment produced nothing.

**Method, reusable:** build the meter, calibrate against one known trigger on
the target checkpoint, then search whatever generated family you like. Each
finetune needs its own search; the signal transfers, the trigger does not.

### Higher-margin variant, and the shape of the phase axis

`trigger_typhoon_c70_dx37_dy35_1350x1800.png` (md5 `d8846499…`) carries more
fp16 headroom than the original half-phase board and is the **preferred**
reproducer. Both are generated by `gen_typhoon_trigger.py` alongside the
healthy phase-0 control.

| variant | headroom | max abs | stock 0.33.0 | stock 0.32.15 | fork, gate on |
|---|---|---|---|---|---|
| **dx=37, dy=35** | **1.068** | 69,940 | `XXXXX` | `XXXXX` | — |
| dx=35, dy=35 | 1.006 | 65,919 | `XXXXX` | `XXXXX` | `HHHHH` |
| dx=0, dy=0 (control) | 0.919 | 60,178 | healthy | — | — |

Phase is a **sharp** axis, not a broad basin — at c=70, dx=36 gives 1.028 and
dx=38 gives 0.999, so the optimum is one or two pixels wide. Check size is
equally spiky. Neither would be found by sampling; both fell out of the meter
in a few minutes each. Every dy neighbour of the optimum also scores lower,
so (37,35) is a genuine local peak, confirmed over 90 further measurements.

### 0.7.1 — a clean negative, and why the method does not reach it

**0.7.1 runs the Go engine.** It has no clip graph and ships no `libmtmd`, so
the meter cannot be loaded there at all — a fact about the *instrument*, not
about exposure. 0.7.1 **is** affected; its equivalent op is `blk.31`
`mlp.down_proj` via `nn.Linear.Forward` → `Mulmat`, and its summation order
differs, so its trigger set differs too.

Screened **98 generated patterns** — check size 14…112 across three geometries
kept under 0.7.1's ~1,003,520 px cap so the pattern reaches the tower
unresized, both at phase 0 and at the half-phase offset that unlocked typhoon,
plus finer phases at dx = c/2 ± 2. Benign control `HHH` first, corpus trigger
`XXX` last:

**0 of 98 fire.** With the 199 synthetics of waves 1–5, that is **297
generated images and no synthetic trigger for 0.7.1's Go engine**, while its
corpus trigger reproduces reliably.

A first pass with the positive control placed *first* reported 26 candidates —
which were exactly the first 26 files alphabetically, one contiguous group,
with every later group clean. That is the poisoned slot draining, not a
finding. Re-running with the control last returned zero. **Put the positive
control last.**

Reaching 0.7.1 properly needs a Go-engine meter — instrumenting `Mulmat` for
`blk.31 mlp.down_proj` and swapping the Go binary into the 0.7.1 image, with
the native payload untouched. Not attempted here.

### A Go-engine node meter for 0.7.1 (`go-engine-node-meter-071.patch`)

0.7.1 cannot load the `libmtmd` meter, so this is its equivalent for the Go
engine: capture block N's `ffn_down` output in `VisionMLP.Forward`, then
compute and report magnitude statistics from `EncodeMultimodal`. Enabled with
`OLLAMA_VISION_METER_LAYER=31`.

**It is a Go-only binary swap** — cgo pulls ggml *headers* only and the CUDA
backend is loaded at runtime from `/usr/lib/ollama`, so the image's native
payload is untouched. Build against a **glibc 2.31** toolchain
(`golang:1.24-bullseye`); 0.7.1's image is Ubuntu 20.04 and a bookworm/trixie
Go image produces a binary that dies on `GLIBCXX_3.4.29 not found`.

**Read it on the CPU backend.** 0.7.1's vendored ggml has no
`GGML_CUDA_CUBLAS_COMPUTE_TYPE` knob — `cu_compute_type` is hardcoded
`CUBLAS_COMPUTE_16F` — so f32 accumulation cannot be forced on CUDA without a
full CUDA rebuild. The CPU backend accumulates in f32 and gives clean
magnitudes. Calibration on 0.7.1's own tower:

| image | engine verdict on 0.7.1 | headroom (CPU) |
|---|---|---|
| corpus `04431b0d` | **X** | **1.032** (max 67,602) |
| plain certificate | H | 0.633 (max 41,486) |

Same clean separation as the clip meter, so the instrument is sound.

**CUDA readback — diagnosed and fixed.** `Context.Compute` calls
`ggml_backend_sched_reset()` immediately after dispatching the async compute,
which releases the graph allocator's intermediate buffers. On CPU the host
allocation still holds the values; on CUDA the binding is dead and
`ggml_backend_tensor_get` returns a correctly-sized slice of zeros. Model
outputs such as logits are unaffected because they live in a persistent buffer.

The fix is to copy the captured tensor into an `Input()`-allocated destination
before computing — that buffer survives the reset and is readable on both
backends — which is the same idiom `ml.Backend` already uses for its own f32
conversion:

```go
dst := ctx.Input().Empty(ml.DTypeF32, t.Shape()...)
dst = t.Copy(ctx, dst)
ctx.Forward(dst).Compute(dst)
```

**On GPU the meter costs ~10 s per image against 155 s on CPU — 15x faster**,
so a 98-pattern sweep is ~15 minutes rather than ~6 hours.

One trap when scraping the output: the warm-up/`Reserve()` pass emits its own
`GO_NODE_STATS` line with a different element count (6,272,000 vs 4,945,920 for
a real 1176x644 image). Read only the lines a request actually produced, or the
warm-up zeros get reported as the measurement.

**On GPU, `n_bad` is a direct trigger detector.** Under fp16 accumulation the
overflow appears as a non-finite element in the node itself — the corpus
trigger reports `n_bad=1` — so the meter detects the fault in the vision graph
without going near the language model. That makes it **immune to slot
poisoning**, unlike an H/X verdict:

| image | GPU `n_bad` | GPU headroom | CPU headroom |
|---|---|---|---|
| corpus trigger | **1** | 0.816 | **1.032** |
| plain certificate | 0 | 0.629 | 0.633 |
| best checkerboard | 0 | 0.706 | 0.782 |

GPU headroom reads lower than CPU for the trigger because the overflowing
element becomes `inf` and is excluded from the finite maximum — `n_bad` is the
signal there, and CPU headroom is the continuous one.

#### Metered result: the checkerboard family is the wrong family for 0.7.1

Ten patterns spanning check size and phase, metered on 0.7.1's own tower
(`results-071-go-meter.txt`):

| image | headroom | engine |
|---|---|---|
| corpus trigger | **1.032** | X |
| best checkerboard (`c70`, phase 0) | **0.782** | H |
| `c70` half-phase | 0.730 | H |
| plain certificate | 0.633 | H |
| worst (`c28` half-phase) | 0.418 | H |

The best member of the family reaches **0.78 against the trigger's 1.03 — a
32 % gap**, and nothing in the size/phase slice closes it. Two things follow.

**Phase inverts here.** On the Go engine `c70` phase 0 (0.782) *beats* `c70`
half-phase (0.730); on the clip path the half-phase variant is the trigger and
phase 0 is healthy. Different summation order, different trigger set — now
measured rather than assumed.

**The blind sweeps were not unlucky, they were mis-aimed.** 297 generated
images found nothing on 0.7.1 because this family tops out ~32 % short, not
because the search was too small. More checkerboard refinement is wasted
effort; a different generative family is needed, and the meter can now rank
candidates instead of guessing — at ~155 s each until the CUDA readback is
fixed.

## FOUND: a public trigger for 0.7.1's Go engine

`trigger_071_nasa_contrast15.png` — NASA image `20040421_exp9_02` ("control
room", **public domain**), contrast-boosted x1.5 about mid-grey and fitted
under 0.7.1's px cap. `gen_071_trigger.py` reproduces it byte-identically from
the shipped base (md5 `0d892db9…`).

The **unmodified base photo is healthy**, so the two ship together as a paired
control differing by one multiplier.

### Evidence — stock `ollama/ollama:0.7.1`, unmodified binary, no env vars

| step | cell | n | result |
|---|---|---|---|
| 1 | benign certificate, virgin slot | 3 | `HHH` |
| 2 | **NASA base, no gain** | 3 | `HHH` |
| 3 | **NASA x contrast 1.5** | 5 | **`XXXXX`** |
| 4 | NASA x contrast 1.5 x scale 1.4 | 5 | **`XXXXX`** |
| 5 | corpus trigger (positive control, **last**) | 3 | `XXX` |

Both candidates fired before the control ran, so contamination cannot explain
them. Meter agreement: `n_bad=6` and `n_bad=5` against the corpus trigger's
`n_bad=1`.

### How the search got there, after 297 blind failures

With the meter reading 0.7.1's own tower at ~10 s/image, three families were
swept and ranked (`results-071-family-search.txt`):

| family | n | best headroom | overflow |
|---|---|---|---|
| checkerboards (size x phase) | 98 | 0.706 | none |
| sinusoids, 1/f noise, gradients, flat colour | 60 | 0.685 | none |
| **public-domain photographs** | 104 | **0.950** | none |
| **photographs + contrast gain** | 50 | **0.985** | **3 hit** |

Every synthetic family plateaus around 0.69–0.71 no matter how it is
parameterised; photographs reach 0.95 straight away. **Gain amplifies structure
an image already has and cannot create it** — the same contrast dial that did
nothing for flat synthetics (0.721 -> 0.731 on the clip path) carried a
photograph over the cliff here.

That also explains the earlier negative honestly: 297 generated images found
nothing not because the sweep was small, but because the entire synthetic
region sits ~30 % short. The meter turned a blind lottery into three families
ranked in under an hour.

**Per-engine summary.** The clip path (0.30+) is triggered by a generated
checkerboard and typhoon by its half-phase variant; 0.7.1's Go engine is
triggered by neither and needs a photograph. Different summation order,
different trigger set — every fold needs its own search, and the meter is what
makes each one cheap.

### Blast radius: this one is cross-engine

Unlike the checkerboards, which are engine-specific, the NASA contrast trigger
fires **both** vision implementations. Base photo first in every cell, n=5 on
the trigger, image identity verified per container after an earlier run was
silently answered by a stale container on a clashing port:

| build | path | model | base | trigger |
|---|---|---|---|---|
| stock 0.7.1 | Go engine | `qwen2.5vl:3b` | `HHH` | **`XXXXX`** |
| stock 0.30.0 | clip | `qwen2.5vl:3b` | `HH` | **`XXXXX`** |
| stock 0.33.0 | clip | `qwen2.5vl:3b` | `HH` | **`XXXXX`** |
| stock 0.33.0 | clip | `typhoon-ocr1.5-3b` | `HH` | `HHHHH` |
| stock 0.33.0 | clip | `qwen2.5vl:7b` | `HH` | `HHHHH` |
| stock 0.7.1 | Go engine | `typhoon-ocr1.5-3b` | `HHH` | `HHHHH` |
| stock 0.7.1 | Go engine | `qwen2.5vl:7b` | `HHH` | `HHHHH` |
| **fork sync-0.33.0, gate on** | clip | `qwen2.5vl:3b` | `HH` | **`HHHHH`** |
| fork sync-0.33.0, gate on | clip | `typhoon` | `HH` | `HHHHH` |

So it spans the 0.7.1 Go engine *and* the 0.30+ clip path — one image covering
both implementations, which no other reproducer we have does. It stays specific
to `qwen2.5vl:3b`: the 7B and the typhoon finetune are clean on **both** engines,
each having its own trigger set. The `qwen25vl` f32 gate heals it.

The 0.7.1 rows for typhoon and the 7B needed a second run. The first used the
corpus image as the control, which only ever triggered `qwen2.5vl:3b`, so it
read `HHH` and left the container with **no control that had actually fired** —
two uninterpretable negatives. Re-run with `qwen2.5vl:3b` in the same container
(trigger `XXXXX`, corpus `XXX`), the detector is demonstrably live and the
negatives stand. A control that cannot fire is not a control.

Read those two rows as *unaffected by this image*, not as unaffected by the
bug: both are `qwen25vl`-family and gate-covered, and neither has been searched
for a trigger of its own on the Go engine.

`gemma4:12b-nvfp4` returned request errors on stock ollama rather than a
verdict, so this run says nothing about it either way; its structural immunity
(zero F16 vision matmuls) rests on the earlier static audit, not on this table.

**Practical read:** an ordinary contrast adjustment of an ordinary press
photograph reaches the fault on every affected release tested, across two
independent engine implementations. Nothing about the input is adversarial or
unusual.

### The triggers do not transfer to mlx-cuda (nvfp4)

`gemma4:12b-nvfp4` on the fork's MLX-CUDA runner, num_ctx pinned to 8192,
n=3 per image:

| image | gemma4:12b-nvfp4 (MLX) | qwen2.5vl:3b (ggml, same container) |
|---|---|---|
| base photo | `HHH` | `HHH` |
| NASA contrast trigger | `HHH` | **`XXX`** |
| typhoon half-phase (dx37) | `HHH` | **`XXX`** |
| checkerboard | `HHH` | **`XXX`** |

Device placement verified rather than assumed: 7,818 MiB resident on GPU-0
under `/usr/bin/ollama`, with `MLX engine initialized ... device=gpu` and
`starting mlx runner subprocess model=gemma4:12b-nvfp4` in the log. A first
reading suggested MLX had not initialised on a GPU at all — that was an
artefact of sampling `nvidia-smi` after container teardown. First request is
~10 min of cold JIT, consistent with the known MLX cold-start behaviour.

**Read this as "these three images do not transfer", not "MLX is immune."**
There is no MLX positive control — no image is known to break it — so the
detector's liveness on the MLX path is unproven, and the ggml control only
shows the harness works. The mechanism also argues against transfer a priori:
the fault is fp16 *accumulation* in cuBLAS on f16-weight GEMMs, while MLX uses
its own kernels and nvfp4 is not fp16 accumulation at all. gemma4 additionally
has zero F16 vision matmuls even on the clip path.

Establishing whether the class exists on MLX needs an MLX-side meter and a
search of its own, exactly as each ggml engine did.

**Output quality checked, not just the verdict.** The H/X verdict alone was
weak evidence here: the degenerate check requires >= 10 characters, so an
*empty* response also scores `H`. Re-run capturing the text, with `think`
explicit and `num_ctx` pinned to 8192:

| image | wall | prefill | decode | response |
|---|---|---|---|---|
| NASA base | 710 s (cold JIT) | 2.4 t/s | 0.1 t/s | correct: "large, high-tech control room… world map… people working at desks" |
| NASA trigger | 6.8 s | 210 t/s | 33.8 t/s | correct: "control room with a large screen displaying a world map…" |
| checkerboard | 2.1 s | 959 t/s | 25.9 t/s | correct: "A black and white checkered pattern covers the entire image." |

All three are accurate descriptions, so MLX genuinely processes these images
rather than merely avoiding a degenerate decode. Warm decode 26-34 tok/s; the
710 s first request is the known MLX cold JIT, not a fault.

**Trap:** with `think` on and `num_predict=120`, the whole budget goes to
reasoning (457 chars) and the response comes back **empty** with
`done_reason=length` — which a naive detector scores as healthy. Pin
`num_predict` high enough for an answer *after* thinking, or disable thinking,
and always look at the text before trusting a verdict.

**Side finding:** the higher-margin typhoon variant (dx=37) also fires
`qwen2.5vl:3b` (`XXX`), where the dx=35 variant did not (`HHHHH`). More
headroom bought a broader trigger, not just a more reliable one.


---

## What this directory keeps, and what it does not

Kept: the shipped trigger artifacts and their generators, the provenance and
rights note, the metered measurements that support the live conclusions
(`results-engine-meter.txt`, `results-071-family-search.txt`,
`results-071-go-meter.txt`), the blind-sweep results that establish the
negatives, `DTYPE-TRACE.md` for the node localisation, `fp16_audit.py` for the
static weight ranking, and `go-engine-node-meter-071.patch` for the pre-0.30
Go engine.

Removed during consolidation: the Hugging Face proxy-screening tooling and its
~115 KB of measurements. That method was **falsified** — on typhoon's own tower
it scored a confirmed trigger at 0.602 and a confirmed non-trigger at 0.975 —
and the finding is what matters, not the raw rows. The clip meter moved to
`llama/compat/801-clip-node-stats-meter.patch`, so the copy here and the
earlier NaN-only tracer it superseded are gone.
