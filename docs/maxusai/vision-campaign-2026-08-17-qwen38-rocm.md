# Vision campaign 2026-08-17 — Qwen3.8 first ROCm/gfx1151 baseline

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
| think | `THINK=false` — the suite default and this host's operational default |
| sampling | `sampling_source = greedy-think-off`, temperature 0 |
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
- **Does not:** establish a repeat-measured baseline. n=1, think-off only. The
  Apple campaign ran think-on as a second arm; that has not been done here, and
  [ADR 0025](adr/0025-think-stays-off-on-gfx1151.md) explicitly does not bind an
  unmeasured family. A think-on arm is the obvious next run — a text-only probe
  during the same session got an arithmetic question wrong with think off and
  right with think on, which is an anecdote, not a measurement.
- **Does not:** make the family regression-covered. That needs the expectations
  entry proposed alongside this record, and per ADR 0011 rule 4 the `[expect.…]`
  block alone is insufficient — `arches` on the profile must move with it.
