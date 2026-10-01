# Vision campaign 2026-09-30 — the nvfp4 fleet and three GGUF tags on an H100 (sm_90)

One host, one build. A GCP `a3-highgpu-1g` VM — one H100 SXM5 80GB (sm_90), collab label
`gcp-a3-highgpu-1g/cuda` — serving natively on `:11434` (a systemd service, restarted cold before
every cell through `RESTART_CMD`). Build **`0.34.4-dynres-0-gb43ee8e`**, a native build of
`v0.34.4-dynres` with the CUDA 13 payloads: llama.cpp `161755f29`, MLX `59d600b`. Tags are
unprefixed, driven from `v0.34.4-dynres` through `run_engine_compare.sh`. Wall clock
2026-09-30 21:25:54 → 22:00:14 think-off and 22:00:31 → 22:40:56 think-on, both `rc=0`.

**What it is.** The first vision-suite campaign on sm_90: the
[2026-09-18 campaign's](vision-campaign-2026-09-18-mlx8a7ba949-nvfp4.md) five nvfp4 tags on the
MLX engine and three GGUF tags on llama.cpp, think-off, and think-on for the two tags that
campaign ran think-on. The request is that campaign's, byte for byte: `prompt_sha` and
`images_sha` equal its values on all 270 arms, so
[its request, verbatim](vision-campaign-2026-09-18-mlx8a7ba949-nvfp4.md#the-request-this-campaign-sends)
is this one's.

**Not analysed here: the differences from other hosts.** Hopper selects different cuBLAS kernels
from sm_120 and from MLX on Metal, and four of the five nvfp4 tags are not the artifacts the
2026-09-18 campaign measured (below). This document records the H100's numbers.

Committed data, assembled from the run captures by script (ADR 0012 rule 8):

- `vision-suite/bench-runs/vision-campaign-2026-09-30-h100-sm90.json` — every cell, both finetext
  arms, and three reruns of one scene cell
- `vision-suite/bench-runs/vision-campaign-2026-09-30-h100-sm90.host-profile.json` — the machine,
  in the `host-profile/1` format: an H100 SXM5 on a PCIe Gen4 x16 link (the GPU supports Gen5),
  Sapphire Rapids, 26 vCPUs, 230 GiB

**Checkpoints measured**, identified by manifest digest (SPEC H17):

| checkpoint | manifest sha256 | config | size |
|---|---|---|---|
| `gemma4:12b-nvfp4` | `ded7a27350032202` | `f6127828935e` | 7.2 GiB |
| `gemma4:26b-nvfp4` | `f60799545325362b` | `3a77d9b825b3` | 15.0 GiB |
| `gemma4:31b-nvfp4` | `a22a363052da7703` | `b72c5344f12d` | 18.1 GiB |
| `qwen3.8:27b-nvfp4` | `5642e97495e1a088` | `25a98d24af80` | 16.9 GiB |
| `qwen3.6:35b-a3b-nvfp4` | `e92a3e94bbca90a8` | `79866ce67dc0` | 22.0 GiB |
| `gemma4:31b-it-q4_K_M` | `17ba34c06c801ba2` | `094d707563aa` | 19.0 GiB |
| `qwen3.8:27b-q4_K_M` | `25b843619e944cd0` | `492b2922d38e` | 16.5 GiB |
| `nemotron3:33b-q8` | `74d89c84a4432530` | `655f279147f2` | 34.0 GiB |

Of the five nvfp4 tags, only `qwen3.8:27b-nvfp4` is the artifact the 2026-09-18 campaign measured
(`5642e97495e1a088`). The other four are different artifacts under the same tags:
- **That campaign's store was already behind the registry** for the three gemma4 tags on the day it
  ran (the store audit in [ocrbench-quantisation-ladder.md](ocrbench-quantisation-ladder.md),
  2026-09-18).
- **`gemma4:12b-nvfp4` and `gemma4:31b-nvfp4` here** are the artifacts the registry served that day.
- **`gemma4:26b-nvfp4` has been re-published again since:** the registry served `f0fc7e0ae494` that
  day.
- **`qwen3.6:35b-a3b-nvfp4` differs too,** and this document does not say why.

**Scope.** Think-off runs all eight tags. Think-on runs only `gemma4:31b-nvfp4` and
`qwen3.8:27b-nvfp4`, the 2026-09-18 campaign's narrowing, so the two stay comparable.

## 1. Think-off, eight tags

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 16384 | **0.727** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:26b-nvfp4 | **MLX** | 16384 | **0.974** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.965** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | **0.999** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384 | **0.966** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 0.965 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 0.977 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| nemotron3:33b-q8 | GGUF | 16384 | 0.878 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 4/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 16384 | 4 | 4 | 3 | 2 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 541 | 86 | 2689 | 6.9 | 520 |
| gemma4:26b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 535 | 92 | 2221 | 6.6 | 547 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 537 | 40 | 1788 | 14.3 | 252 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 549 | 45 | 7014 | 12.7 | 284 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 1 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 534 | 71 | 6173 | 8.0 | 451 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 538 | 151 | 1421 | 4.8 | 757 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 2 | 1 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 544 | 78 | 1646 | 8.5 | 421 |
| nemotron3:33b-q8 | GGUF | 16384 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 244 | 4777 | 2.7 | 1355 |

Provenance (from score files): host(s) http://127.0.0.1:11434 · build(s) 0.34.4-dynres-0-gb43ee8e · think=false

## 2. Think-on, the two cells the 2026-09-18 campaign ran

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.965** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | **0.994** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Gen tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1982 | 41 | 1751 | 49.8 | 72 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 1 | 0 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 997 | 45 | 7049 | 22.6 | 160 |

Provenance (from score files): host(s) http://127.0.0.1:11434 · build(s) 0.34.4-dynres-0-gb43ee8e · think=on

No cell reached its `num_predict` (8192 at the 16384 rung), so no think-on number is a cap.

## 3. The 12b scene cell is stable

`gemma4:12b-nvfp4`'s scene IoU, 0.727, is the lowest of the MLX rows. Re-run alone, three times,
each after a cold restart (2026-10-01), it reproduces: the same 541-token answer in pixel space,
with every box, label and colour found every time.

| run | start | Scene bbox IoU | coordinate space | boxes · labels · colours | answer tokens |
|---|---|---|---|---|---|
| campaign cell | warm | 0.727 | pixel/xyxy | 6/6 · 6/6 · 6/6 | 541 |
| rerun 1 | restart_cmd | 0.724 | pixel/xyxy | 6/6 · 6/6 · 6/6 | 541 |
| rerun 2 | restart_cmd | 0.727 | pixel/xyxy | 6/6 · 6/6 · 6/6 | 541 |
| rerun 3 | restart_cmd | 0.728 | pixel/xyxy | 6/6 · 6/6 · 6/6 | 541 |

## 4. What the tables show

- **`qwen3.8:27b-nvfp4` grounds best:** scene IoU 0.999 think-off and 0.994 think-on, the only
  MLX tag at 5/5 `name_bbox` in-band.
- **The gemma4 26b and 31b tags read the most fine text,** `[4,4,4,4,3]`, on both engines.
- **On this H100, llama.cpp decodes `gemma4:31b` 3.8× faster than MLX** — 151 against 40 gen tok/s,
  4.8 against 14.3 s/req — with the same grounding, extraction and fine-text cells. The two rows
  are different artifacts (`q4_K_M` against `nvfp4`), so this compares the engine and the format
  together.
- **`nemotron3:33b-q8` is the fastest cell** (244 gen tok/s, 2.7 s/req) and the only one to miss
  the serial.
- **Thinking changes little but time.** `gemma4:31b-nvfp4` scores the same cells either way;
  `qwen3.8:27b-nvfp4` loses 0.005 of scene IoU, one 9 px hit and the `q4_bbox_hit` multi-image
  question — as it did think-on in the 2026-09-18 campaign — at 1.8–3.5× the time per request.

## 5. Limits

- One run per cell, except the three scene reruns: no repeats control for the rest.
- `powermode` does not apply on CUDA.
- The MLX and GGUF rows of one model are different artifacts, so an engine comparison here is
  also a quantisation-format comparison.
- Cross-host comparison is out of scope (above), and for four of the five nvfp4 tags it would
  also compare different weights.

## Reproducing

The tables render from the bundle alone, identically, with `summarize_engine_compare.py --bundle`
(#432). The loop renders each scene rerun; their IoUs are §3's.

```bash
cd docs/maxusai/vision-suite
B=bench-runs/vision-campaign-2026-09-30-h100-sm90.json
python3 summarize_engine_compare.py --bundle $B --think false gemma4:12b-nvfp4 gemma4:26b-nvfp4 \
  gemma4:31b-nvfp4 qwen3.8:27b-nvfp4 qwen3.6:35b-a3b-nvfp4 gemma4:31b-it-q4_K_M qwen3.8:27b-q4_K_M \
  nemotron3:33b-q8
python3 summarize_engine_compare.py --bundle $B --think on gemma4:31b-nvfp4 qwen3.8:27b-nvfp4
for n in 1 2 3; do
  python3 summarize_engine_compare.py --bundle $B --think false --prefix rerun_${n}_ \
    --expect scene_single gemma4:12b-nvfp4
done
```
