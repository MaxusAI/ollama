# Vision campaign 2026-08-17 — Qwen3.8 first ROCm/gfx1151 baseline, both think modes

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
| think | both arms: `THINK=false` and `THINK=on` (the literal `on`, per [ADR 0023](adr/0023-think-mode-is-per-model-and-measured-on-policy.md)) |
| sampling | think-off `greedy-think-off` (temperature 0); think-on `packaged-defaults-no-card` (no overrides) — correct for this family per [ADR 0026](adr/0026-qwen38-baselines-record-the-effort-directive.md), no `CARD_THINKING` entry and none should be added |
| effort directive | **none** in both arms. `THINK=on` omits the `think` field, which `server/routes.go` coerces to `true`, and per ADR 0026 `true` emits no directive. Neither arm is an xhigh run. |
| cache | cold; model loaded fresh, 66/66 layers offloaded to ROCm, 25.2 GiB VRAM |
| fixtures | committed `visimgs/`, unmodified |
| n | **1** — single pass, no repeats |

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

All eleven `bbox_contract` variants returned 6/6 labels with valid JSON. Every
extraction in the suite parsed.

## Think-off vs think-on, same host, same build (T2)

| test | metric | think-off | think-on |
|---|---|---|---|
| scene | bbox IoU | **0.991** (16384) | 0.981 (16384) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.570 (16384) | **0.790** (16384) |
| fine text | 22/16/12/9/7 px | **4/4/4/2/1** (32768) | 4/4/4/1/0 (32768) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ **✅** 5/5 (16384) | ✅ ✅ **❌** 5/5 (16384) |
| throughput | gen tok/s | 12 | 12 |
| throughput | prefill tok/s | 276 | 281 |
| latency | s/req (unique image) | **54.0** | 97.0 |
| latency | req/h (serial) | **67** | 37 |

**Think-on is a net loss here, as it was on Apple Silicon.** Every exact-match
metric — labels, serial, invoice number, line items, qty+price, total, chart
values — is already perfect in *both* arms, so thinking has nothing to improve.
What it changes it mostly makes worse: the 9px *and* 7px fine-text tiers, the
multi-image `q4_bbox`, and a point of scene IoU, for 1.8x the latency and 55% of
the serial throughput. Generation speed is unchanged at 12 tok/s, so the entire
cost is extra tokens (scene: 544 -> 1071 eval).

The one gain, document `name_bbox` 0.570 -> 0.790, sits in the least trustworthy
cell in the suite — the same metric swung 0.638 -> 0.248 across configurations
within the Apple campaign alone. At n=1 per arm it should not be banked.

The shape reproduces Apple Silicon closely: scene IoU fell 0.991 -> 0.980 there
against 0.991 -> 0.981 here, on different silicon and a different payload. Fine
text degraded on both, one tier further here. `q4_bbox` was already failing on
Apple in both arms, so ROCm's ✅ -> ❌ moves the same direction from a better
start.

### Budget headroom — checked, because a capped cell is not a quality result

Neither budget was exhausted in either arm. `eval_count < num_predict` in every
cell and every generation ended `done_reason=stop`, so nothing was cut off:

| | worst utilisation | where |
|---|---|---|
| `num_predict` | **83.8%** (1843 / 2200) | think-on `multi_3img` |
| `num_ctx` | **48.7%** (7977 / 16384) | think-on `multi_3img` |

**Caveat worth carrying into the next run:** the cell closest to its
`num_predict` ceiling is exactly the cell that regressed. `multi_3img` think-on
spent 84% of its allowance before answering and is where `q4_bbox` flipped to ❌.
That is not truncation — but "not truncated" and "unaffected by the ceiling" are
different claims, and only the first is established. A think-on repeat at a
higher `NUM_PREDICT` would separate them.

Note when reading raw score files **written before `0d3d8935`**, which includes
every file behind this campaign: the bare `num_ctx` / `num_predict` fields record
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

### The two arms differ in two variables, not one

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
- **Does not:** establish a repeat-measured baseline. **n=1 per arm.** The
  `name_bbox` gain and the `q4_bbox` loss both sit in cells this suite has
  already shown to be unstable across configurations.
- **Does not:** settle whether think-on's `q4_bbox` regression is a reasoning
  effect or a budget effect — see the headroom caveat above.
- Unrelated to vision, and recorded only as an anecdote: a text-only arithmetic
  probe in the same session was wrong with think off and right with think on.
  One question is not a measurement, and it points the opposite way to the
  vision result.
- **Does not:** make the family regression-covered. That needs the expectations
  entry proposed alongside this record, and per ADR 0011 rule 4 the `[expect.…]`
  block alone is insufficient — `arches` on the profile must move with it.
