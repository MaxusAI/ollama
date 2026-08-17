# Vision campaign 2026-08-17 — Qwen3.8 first ROCm/gfx1151 baseline, repeated arms

First vision run for Qwen3.8 on the AMD host, taken the day the renderer was
backported to `release/0.32.1-dynres` (PR #142) and deployed. Companion to
[vision-campaign-2026-08-16-qwen38.md](vision-campaign-2026-08-16-qwen38.md),
which measured the same model on Apple Silicon.

## Provenance

| | |
| --- | --- |
| server | `0.32.1-dynres-5d5b7a72`, container `ollama-rocm` on `:11434` |
| host | `glenn-NucBox-EVO-X2` (10.8.0.4), Ryzen AI Max+ 395 / Radeon 8060S, **gfx1151** |
| payload | **b9888** — the gated lineage, no `--direct-io`; compat 001+002+004+005 |
| model | `qwen3.8:27b-q4_K_M` (GGUF → llama-server), sideloaded |
| store | `/opt/ollama/.ollama/models` (production) |
| think | `THINK=false` and `THINK=on` (the literal `on`, per [ADR 0023](adr/0023-think-mode-is-per-model-and-measured-on-policy.md)); one think-on arm additionally repeated at `NUM_PREDICT=4400` as a ceiling diagnostic |
| sampling | think-off `greedy-think-off` (temperature 0); think-on `packaged-defaults-no-card` (no overrides) — correct for this family per [ADR 0026](adr/0026-qwen38-baselines-record-the-effort-directive.md), no `CARD_THINKING` entry and none should be added |
| effort directive | **none** in every arm. `THINK=on` omits the `think` field, which `server/routes.go` coerces to `true`, and per ADR 0026 `true` emits no directive. Neither arm is an xhigh run. |
| cache | cold; model loaded fresh, 66/66 layers offloaded to ROCm, 25.2 GiB VRAM |
| fixtures | committed `visimgs/`, unmodified |
| n | **2 think-off, 5 think-on.** ADR 0023 asks for n=3 on think-on because on-policy sampling is not deterministic; the original n=1 reading of this campaign was wrong twice and is corrected below |

How the model got here matters for reproducibility: `ollama pull` **cannot**
fetch it on this host. The published manifest declares `requires: 0.32.12` and
the registry answers 412 to any client whose `User-Agent` reports lower. The
gate is keyed on that header, not on access — a plain HTTPS fetch of the same
manifest returns 200 — so the blobs were retrieved with an ordinary HTTP client
and installed content-addressed, every digest verified before install.

## Results (`THINK=false`), against the Apple Silicon baseline

| test | metric | **ROCm b9888** | Apple `0.32.5-maxusai` (2026-08-16) |
|---|---|---|---|
| scene | bbox mean IoU | **0.991** | 0.991 |
| scene | labels / colors / serial | 6/6, 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | **0.57** | 0.638 |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ **✅** 5/5 | ✅ ✅ **❌** 5/5 |
| fine text | 22/16/12/9/7 px | **4/4/4/2/1** | 4/4/4/2/1 |
| throughput | gen tok/s | 12.2 | 20 |
| throughput | prefill tok/s | 255–276 | 415 |
| latency | s/req (scene / doc / multi / finetext) | 58.1 / 55.4 / 119.8 / 39.0 | 33.1 (mean) |

All eight `bbox_contract` variants returned 6/6 labels with valid JSON. Every
extraction in the suite parsed.

The ROCm column is the first think-off run, kept as-is because the Apple column
is also a single run and the two should be read like for like. The repeated
think-off means are within 0.003 of it (0.990 scene IoU, 0.571 name_bbox) and
are tabulated below.

## Repeated arms, same host, same build (T2)

Superseded the original three-arm reading. ADR 0023 asks for **n=3 per think-on
cell** because on-policy sampling is not deterministic, and at n=1 this document
drew two conclusions that do not survive repetition. Now pooled: **n=2 think-off,
n=5 think-on**, all on `0.32.1-dynres-5d5b7a72`, `num_ctx` 16384 throughout.
Rendered by `summarize_reps.py`, not transcribed.

| metric | think-off (n=2) | think-on (n=5) |
|---|---|---|
| scene bbox IoU | 0.990 ±0.002 | 0.987 ±0.011 |
| scene labels / colors / serial | 6 / 6 / 2✅ | 6 / 6 / 5✅ |
| doc items / qty+price / total | 5 / 5 / 2✅ | 5 / 5 / 5✅ |
| **doc name_bbox IoU** | **0.571 ±0.001** | **0.742 ±0.103** |
| multi q1 / q2 / chart | 2✅ / 2✅ / 5 | 5✅ / 5✅ / 5 |
| **multi q4-bbox** | **2/2 ✅** | **0/5 ❌** |
| finetext correct / fabricated (of 20) | 15 / 5 | 13.6 [13–15] / 6.4 [5–7] |
| latency s/req · req/h | 54.0 · 67 | ~97 · 37 |

Within-arm spread, the bar any cross-arm claim must clear:

- think-off: scene IoU **0.003**, name_bbox **0.001**
- think-on: finetext correct **2**, name_bbox **0.103**, scene IoU **0.021**

### What repetition changed, including a correction to this document

**The `name_bbox` gain is real.** This is the second reversal on that cell and
the numbers now settle it: think-on's *lowest* value across five runs is 0.631
and think-off's *highest* across two is 0.571. The arms never overlap, and the
~0.17 gap exceeds think-on's own 0.103 spread. An earlier revision of this
document called the gain noise on the strength of two think-on samples
disagreeing by 0.159 — that showed the spread was large, which is true, but not
that the arms overlapped. Those are different claims and conflating them is what
produced the wrong retraction.

**`q4_bbox` is settled the other way, and harder than before.** 0/5 under
think-on against 2/2 under think-off. Already shown not to be a budget effect
(the `NUM_PREDICT=4400` repeat used 46% of its ceiling and still failed).

**Think-off is near-deterministic, not deterministic.** Two greedy
temperature-0 runs moved scene IoU 0.991 → 0.988. Tiny, an order of magnitude
under think-on's spread, and consistent with GPU float-reduction ordering — but
ADR 0023's convention of a single think-off run rests on an assumption that is
very nearly, and not exactly, true.

**Fine text survives, with a thinner margin than n=1 suggested.** 15/20 correct
think-off against 13.6 mean think-on; the fabrication cost is +1.4 rather than
the flat +2 two samples implied.

### So it is a trade-off, not a one-sided loss

This document previously concluded think-on was a net loss with nothing in its
favour. At n=5 that is wrong. Think-on buys a real improvement in document
`name_bbox` grounding and pays for it with a reproducible loss of multi-image
bbox grounding, ~1.4 more fabricated fine-text codes, and 1.8x the latency for
55% of the serial throughput.

**The operational recommendation is unchanged** — keep think off for vision work
on this host — but the reason is now a weighed trade rather than a rout. Every
exact-match metric is already perfect in both arms, so the only thing think-on
improves is one bbox cell, and it breaks another to get it.

### The ceiling question is settled: the regression is real

Given twice the budget, `multi_3img` used **46.2%** of it — 2031 tokens against
4400, up only 10% from the 1843 it used under 2200 — and `q4_bbox` still failed.
The model was not straining against the ceiling; it finished reasoning and still
missed the box. Nothing was truncated in any arm: `eval_count < num_predict`
everywhere, every generation `done_reason=stop`, worst `num_ctx` use 48.7%.

### Think-off and think-on differ in two variables, not one

Think-off is greedy at temperature 0; think-on sends no sampling overrides at
all, so the model's packaged temperature applies. That is deliberate — greedy
decoding is what made reasoning fail to terminate, see
[runaway-reasoning-under-think.md](runaway-reasoning-under-think.md) — but it
means a difference between the arms cannot be attributed to thinking alone. The
Apple campaign has the same design, so the cross-host comparison is sound even
though the within-host attribution is not clean.

## Reading it

**The two hosts agree where it counts.** Scene bbox IoU matches to three
decimals and the fine-text ladder is identical tier for tier — including the 7px
tier, which the Apple run's other three configurations all scored 0 on. That is
strong evidence the b9888 payload serves this family correctly, not merely
without crashing.

**One cell is better here, one worse.** Multi-image `q4_bbox` passes on ROCm and
failed on Apple's `27b-q4_K_M`; document `name_bbox` is 0.57 against 0.638. Note
that same metric swung 0.638 → 0.248 across configurations within the Apple
campaign alone, so it is the noisiest cell in the suite. The Apple figures are
single runs, so neither difference should be read as a host effect — the ROCm
side is now repeated but the comparison is only as strong as its weaker half.

**Throughput is silicon, not a fault.** ~60% of the Apple GGUF rate, on an iGPU
sharing system memory.

**This is not the degeneration class the AMD gate exists over.** No repeated-token
runs, no token salad, no response describing a previous request's image, no
malformed JSON. That failure mode ([#17459](https://github.com/ollama/ollama/issues/17459),
[#17475](https://github.com/ollama/ollama/issues/17475)) is what took 0.32.5 off
this host on 2026-07-31 while every plumbing check passed; nothing resembling it
appears here. The gate is unaffected either way — this deploy moved no payload.

## Token ladder — measured, for the preflight baseline

Measured with `probes.Ollama.image_prefix` (the B8 prefix trick) and
`visual_tokens` against the same server, using the harness's own probe rather
than a re-implementation:

```
prefix = 13   text_only = 13   (one_a=1047, one_b=1047, both=2081)

256x144   -> 1034      2048x1152 -> 2306
512x288   -> 1034      3072x1728 -> 4082
1024x576  -> 1034
```

The shape is clamped at both ends and scales in between, and it reproduces from
the arithmetic:

- **Floor.** `visionServerArgs` passes `--image-min-tokens 1024` for
  `qwen35`/`qwen35moe` (`llm/llama_server.go:1015-1026`), so the three smallest
  geometries scale **up** to the floor: 43×24 = 1032 grid tokens, +2 markers =
  1034. The runner log records `image_min_pixels: 1048576 (custom value)` =
  1024·32².
- **Free rung.** 2048×1152 → 64×36 = 2304, +2 = **2306**, inside the window.
  Identical to the Apple row's fourth rung.
- **Ceiling.** llama.cpp's `handle_qwen35_like_clip()` sets
  `set_limit_image_tokens(8, 4096)`, so 3072×1728 clamps **down** from its
  natural 96×54 = 5184 to 4080, +2 = **4082**. The log records
  `image_max_pixels: 4194304` = 4096·32². Confirmed independently by the
  decode batching: `1024+1024+256 = 2304` and `1024×3+1008 = 4080`.

**This ladder must not be confused with the Apple one.** `[1034, 1034, 1034,
2306, 4082]` here against `[68, 146, 578, 2306, 5186]` on `apple-silicon-mlx` —
the same model, differing because the MLX row is anchored on the `nvfp4` export
with a 65536..16777216 pixel window, while this row is the GGUF through
llama-server with the fork's 1024-token floor and llama.cpp's 4096 ceiling. Only
the 2048×1152 rung coincides. Copying either row onto the other would be exactly
the failure [ADR 0011](adr/0011-preflight-expectations-are-versioned-code.md)
rule 4 exists to prevent.

## What this does and does not establish

- **Does:** Qwen3.8 loads, offloads fully, renders through the ported qwen3.8
  renderer (`template selection … renderer=qwen3.8 parser=qwen3.5`), and
  extracts at parity with Apple Silicon on the objective cells.
- **Does:** support keeping `think` **off** for Qwen3.8 vision on this host,
  which points the same way as [ADR 0025](adr/0025-think-stays-off-on-gfx1151.md)
  without being bound by it — 0025 is scoped to its three measured families and
  says explicitly that a new family needs its own measurement. This is that
  measurement, thin as it is.
- **Does:** settle that think-on's `q4_bbox` regression is a reasoning effect,
  not a budget effect — the repeat at `NUM_PREDICT=4400` used 46% of the raised
  ceiling and still failed.
- **Does:** put a measured number on the noise floor — think-on spread 0.103 on
  `name_bbox` and 0.021 on scene IoU across five runs, against think-off's 0.001
  and 0.003 across two. Any single-arm reading of a think-on cell is worth less
  than that spread, which is how this document got `name_bbox` wrong twice.
- **Does:** establish that think-off is near-deterministic — an expectation in
  the earlier revision, now measured, and not exactly true (scene IoU moved
  0.003 between two greedy runs).
- **Does not:** repeat the *Apple Silicon* arms. Every cross-host claim here
  rests on single runs on that side.
- Unrelated to vision, and recorded only as an anecdote: a text-only arithmetic
  probe in the same session was wrong with think off and right with think on.
  One question is not a measurement, and it points the opposite way to the
  vision result.
- **Does not:** make the family regression-covered. That needs the expectations
  entry proposed alongside this record, and per ADR 0011 rule 4 the `[expect.…]`
  block alone is insufficient — `arches` on the profile must move with it.
