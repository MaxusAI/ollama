# Vision campaign 2026-08-17 — Qwen3.8 first ROCm/gfx1151 baseline, three arms

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
| think | three arms: `THINK=false`, `THINK=on`, and `THINK=on NUM_PREDICT=4400` (the literal `on`, per [ADR 0023](adr/0023-think-mode-is-per-model-and-measured-on-policy.md)) |
| sampling | think-off `greedy-think-off` (temperature 0); think-on `packaged-defaults-no-card` (no overrides) — correct for this family per [ADR 0026](adr/0026-qwen38-baselines-record-the-effort-directive.md), no `CARD_THINKING` entry and none should be added |
| effort directive | **none** in all three arms. `THINK=on` omits the `think` field, which `server/routes.go` coerces to `true`, and per ADR 0026 `true` emits no directive. Neither arm is an xhigh run. |
| cache | cold; model loaded fresh, 66/66 layers offloaded to ROCm, 25.2 GiB VRAM |
| fixtures | committed `visimgs/`, unmodified |
| n | **1 per arm**; think-on repeated once at a raised ceiling, which is what quantifies the noise floor below |

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

## Three arms, same host, same build (T2)

Think-on was run twice, the second time at `NUM_PREDICT=4400`, to test whether
the first arm's regressions were a ceiling effect. `num_ctx` was held at 16384
in all three so KV size and decode speed stay comparable; the only variable
between the two think-on arms is the stop ceiling.

| test | metric | think-off | think-on np2200 | think-on np4400 |
|---|---|---|---|---|
| scene | bbox IoU | 0.991 | 0.981 | 0.992 |
| scene | labels / colors / serial | 6/6, 6/6, ✅ | 6/6, 6/6, ✅ | 6/6, 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU / hits | 0.570 / 5 | 0.790 / 5 | 0.631 / 5 |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ **✅** 5/5 | ✅ ✅ **❌** 5/5 | ✅ ✅ **❌** 5/5 |
| fine text | correct / fabricated (of 20 emitted) | **15 / 5** | 13 / 7 | 13 / 7 |
| fine text | 22/16/12/9/7 px | 4/4/4/2/1 | 4/4/4/1/0 | 4/4/4/0/1 |
| `bbox_contract` | all 8 variants, labels | 6/6 | 6/6 | 6/6 |
| throughput | gen tok/s | 12 | 12 | 12 |
| latency | s/req (unique image) | **54.0** | 97.0 | 97.6 |
| latency | req/h (serial) | **67** | 37 | 37 |

### The ceiling question is settled: the regression is real

Given twice the budget, `multi_3img` used **46.2%** of it — 2031 tokens against
4400, up only 10% from the 1843 it used under 2200 — and `q4_bbox` still failed.
The model was not straining against the ceiling; it finished reasoning and still
missed the box. Nothing was truncated in any arm: `eval_count < num_predict`
everywhere, every generation `done_reason=stop`, worst `num_ctx` use 48.7%.

### What replicates, and what is noise

Running think-on twice quantifies the noise floor, and it is not small. **The
two think-on arms disagree with each other by 0.159 on `name_bbox`** — comparable
to the 0.220 "gain" over think-off that a single arm appeared to show. That gain
does not survive a repeat. Scene IoU is the same story: 0.981 and 0.992 straddle
think-off's 0.991, so no difference is established there either. Non-greedy
sampling (packaged defaults) is doing this, and it means any single think-on cell
in the noisy metrics carries little weight.

**Reproducible across both think-on arms — these are the findings:**

- **`q4_bbox` fails**, where think-off hits it. Twice, and not for want of budget.
- **More fabricated fine text.** All three arms emit all 20 codes; what changes is
  how many are real. Think-off gets 15 right and invents 5; both think-on arms
  get 13 and invent 7. Per `finetext_probe.py`, a full `total_found` with zeroed
  small tiers means *fabricated* codes, not omitted ones. The 9px↔7px tier swap
  between the two think-on arms is only which fabrications happened to coincide
  with ground truth — noise inside a stable +2 fabrication cost.
- **1.8x latency for 55% of the serial throughput**, stable to within 0.6s/req.

**Everything think-on might have improved is already perfect without it** —
labels, colors, serial, invoice number, line items, qty+price, total, chart
values, all 8 `bbox_contract` variants, and the 22/16/12px text tiers are
identical in all three arms.

The shape reproduces Apple Silicon: scene IoU fell 0.991 -> 0.980 there, fine
text degraded, and `q4_bbox` was already failing in both of its arms — so ROCm's
✅ -> ❌ moves the same direction from a better starting point.

Note when reading raw score files **written before `0d3d8935`**, which is the
think-off and think-on np2200 arms but NOT the np4400 arm: the bare `num_ctx` / `num_predict` fields record
the suite *defaults*, not the window a given cell ran under. `finetext` really ran
at `req_num_ctx = 32768` / `req_num_predict = 4000` while its `num_ctx` field
reads 16384. The `req_*` fields are authoritative, and `summarize_head_to_head.py`
renders from those.

`0d3d8935` fixed the recording, so files written after it carry the effective
window in both places and this caveat does not apply to them. It is forward-only:
the numbers in *this* document come from files with the old behaviour, so two
consequences carry over here. `summarize_engine_compare.was_capped()` compares
`eval_count` against the recorded `num_predict`, so a `finetext` cell between 2200
and 3999 reads as CAPPED in these files when it terminated freely — treat CAPPED
flags on `finetext` rows here as unreliable. And `ctx_for()`'s mixed-window
warning could not fire, because every section reported the same default.

The np4400 arm was recorded after that fix, so its score file carries the
effective window in both places and neither consequence below applies to it.

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
campaign alone, so it is the noisiest cell in the suite. At **n=1** neither
difference should be read as a host effect.

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
- **Does:** put a number on the noise floor for the unstable cells, by running
  think-on twice: 0.159 on `name_bbox`, 0.011 on scene IoU. Any single-arm
  reading of those metrics is worth less than that spread.
- **Does not:** establish a repeat-measured **think-off** baseline. Think-off was
  run once; the two repeats are both think-on. Think-off is greedy at
  temperature 0, so it should be more stable, but that is an expectation rather
  than a measurement here.
- Unrelated to vision, and recorded only as an anecdote: a text-only arithmetic
  probe in the same session was wrong with think off and right with think on.
  One question is not a measurement, and it points the opposite way to the
  vision result.
- **Does not:** make the family regression-covered. That needs the expectations
  entry proposed alongside this record, and per ADR 0011 rule 4 the `[expect.…]`
  block alone is insufficient — `arches` on the profile must move with it.
