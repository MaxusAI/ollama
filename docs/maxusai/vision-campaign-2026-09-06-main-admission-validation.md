# Vision validation 2026-09-06: `main` after #276 — MLX admission prices the context rung

> Verdict: no regression on the CUDA host. Every think-off cell equals the tagged
> 0.33.3 build and the 0.33.2 baseline on every quality row the templates render;
> preflight is identical to the tagged build; the four arms that OOM'd on Sep 4
> converge 24/24 under headroom and are refused at admission under a cap that
> cannot hold the rung; the automatic-context clamp fires on a real load. One
> single-arm contract flip on qwen3.6 is under an n=3 repeat (section at the end).

## Provenance

- **Date / host:** 2026-09-06, CUDA box (RTX PRO 6000 Blackwell, 97.9 GB), shared
  with the production endpoint on `:11497` throughout. Power mode `n/a`.
- **Build under test:** `maxusai/ollama:main-a523d60b` — the `sync-0.33.3` image
  (payload llama.cpp b10760 `0f3a71be1`, MLX `37c26e57`, patchset 903 + fork band)
  with `main`'s Go binary swapped in (`0.33.3-dynres-10-ga523d60`); the native
  payload is byte-identical to the tagged image, verified with
  `git diff --name-only 0c4f09d4..a523d60b -- x/mlxrunner/mlx MLX_VERSION MLX_C_VERSION CMakeLists.txt cmake/ llama/ ml/ Dockerfile`.
  What `main` adds over the tag: #271 (two-pass structured-output metrics on
  `/api/generate`) and #276 (MLX admission prices weights + KV(`num_ctx`) +
  calibrated headroom; explicit rungs refuse, automatic rungs clamp).
- **References:** `mlx0333cu_1_` (tagged 0.33.3, 2026-09-04) and `mlx0332cu_1_`
  (0.33.2, 2026-09-05), both on this host; the Sep-4 discriminator `mlx0333cuRR_`;
  the 0.32.14 `sync15_1_` cells for the repeats.
- **Endpoint / sampling:** `run_engine_compare.sh` defaults — `/api/chat`, think
  off at temperature 0 (ADR 0029), `num_predict` 2200, ladder from `num_ctx` 8192
  to 65536, one runner, cold restart per cell, `OLLAMA_MAX_LOADED_MODELS=1`,
  **`OLLAMA_GPU_OVERHEAD` 16 GiB** (headroom, not a cap — vision-suite README).
- **Scripts:** `claude-scratch/gate-main276.sh` (steps 1a/1b/2/3a/3b),
  `test-reload-fix.sh` (#279 proof), `repeat-reasoning-arm.sh`; render:
  `render-main276.sh` → every table below is generator output (ADR 0012 rule 1),
  rendered after the campaign completed (rule 8).

## Step 1a — the four Sep-4 OOM arms, three repeats, under headroom

`ONLY_TESTS=multi_3img_anchored,multi_3img,bbox_contract_perobject,bbox_contract_adv_norm1`,
`REPEATS=3`, gemma4:26b and 31b. Driver summary: **OOMs=0 errors=0 suites=6**
(24 arm-runs). The `summarize_reps` tables are in the Repeats section below.

## Step 1b — the rung that cannot fit is refused at admission

Same models under `OLLAMA_MLX_MEMORY_LIMIT` = 35 GiB (runner: `derived 45.7 GiB,
using 35.0 GiB`), text-only requests:

```
  31b@8192  -> 200 667s ok eval=7 prompt_eval=19
  31b@65536 -> 200 0s ok eval=7 prompt_eval=19
  26b@65536 -> 200 838s ok eval=7 prompt_eval=19   (need 32.2 < 35: must admit)
  31b@65536 again (evictable retry path) -> 500 4s model requires 37.6 GiB but OLLAMA_MLX_MEMORY_LIMIT caps the MLX budget at 35.0 GiB
```

31b at 8192 (need 33.2 GiB) is admitted; 31b at 65536 (need 37.6 GiB) is
**refused in 4 s at admission**, naming the knob, where 0.33.3 aborted mid-prefill
with `cudaMallocAsync … out of memory` after minutes. The second line — 31b at
65536 served in 0 s by the runner still warm from 8192 — is the gap #279 closed
the same day (see the proof section).

## Step 2 — five-model think-off campaign, and the same cells on 0.33.3 and 0.33.2

Driver summary: **OOMs=0 errors=0 suites=5**. Tables (generator output):

## T1 — campaign matrix, main (`main276_1_`)

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 8192 | **0.956** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-nvfp4 | **MLX** | 8192 | **0.972** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-nvfp4 | **MLX** | 8192 | **0.962** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.8:27b-nvfp4 | **MLX** | 8192 | **0.999** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 8192 | **0.964** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 8192 | 4 | 4 | 3 | 0 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 535 | 31 | 1016 | 19.1 | 189 |
| gemma4:26b-nvfp4 | **MLX** | 8192 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 541 | 32 | 1116 | 18.6 | 194 |
| gemma4:31b-nvfp4 | **MLX** | 8192 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 537 | 23 | 681 | 26.1 | 138 |
| qwen3.8:27b-nvfp4 | **MLX** | 8192 | 4 | 4 | 4 | 2 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 549 | 16 | 5222 | 35.7 | 101 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 8192 | 4 | 4 | 4 | 2 | 1 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 537 | 19 | 4908 | 28.5 | 126 |

Provenance (from score files): host(s) http://127.0.0.1:11511 · build(s) 0.33.3-dynres-10-ga523d60 · think=false

## T2 — head-to-head, gemma4:12b-nvfp4 think-off: main · 0.33.3 (`mlx0333cu`, Sep 4) · 0.33.2 (`mlx0332cu`, Sep 5). Cross-build by design: the MIXED footer names the three builds; all columns on the CUDA host.

| test | metric | main276_1_gemma4_12b-nvfp4_thinkfalse | mlx0333cu_1_gemma4_12b-nvfp4_thinkfalse | mlx0332cu_1_gemma4_12b-nvfp4_thinkfalse |
|---|---|---|---|---|
| scene | bbox IoU | 0.956 (8192) | 0.956 (8192) | 0.956 (8192) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.714 (8192) | 0.714 (8192) | 0.714 (8192) |
| fine text | 22/16/12/9/7 px | 4/4/3/0/0 (8192) | 4/4/3/0/0 (8192) | 4/4/3/0/0 (8192) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| throughput | gen tok/s | 31 | 26 | 28 |
| throughput | prefill tok/s | 1016 | 626 | 936 |
| latency | s/req (unique image) | 19.1 | 23.0 | 20.8 |
| latency | req/h (serial) | 189 | 157 | 173 |

Provenance (from score files): host(s) http://127.0.0.1:11503, http://127.0.0.1:11506, http://127.0.0.1:11511 · build(s) 0.33.2-dynres-5-g2b95b4a, 0.33.2-dynres.1-39-g242cad5, 0.33.3-dynres-10-ga523d60 ⚠ MIXED — columns are not one campaign

## T2 — head-to-head, gemma4:26b-nvfp4 think-off: main · 0.33.3 (`mlx0333cu`, Sep 4) · 0.33.2 (`mlx0332cu`, Sep 5). Cross-build by design: the MIXED footer names the three builds; all columns on the CUDA host.

| test | metric | main276_1_gemma4_26b-nvfp4_thinkfalse | mlx0333cu_1_gemma4_26b-nvfp4_thinkfalse | mlx0332cu_1_gemma4_26b-nvfp4_thinkfalse |
|---|---|---|---|---|
| scene | bbox IoU | 0.972 (8192) | 0.973 (8192) | 0.972 (8192) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.816 (8192) | 0.816 (8192) | 0.816 (8192) |
| fine text | 22/16/12/9/7 px | 4/4/4/3/3 (8192) | 4/4/4/3/3 (8192) | 4/4/4/3/3 (8192) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (8192) | error | ✅ ✅ ✅ 5/5 (8192) |
| throughput | gen tok/s | 32 | 32 | 28 |
| throughput | prefill tok/s | 1116 | 1091 | 1271 |
| latency | s/req (unique image) | 18.6 | 18.6 | 20.7 |
| latency | req/h (serial) | 194 | 194 | 174 |

Provenance (from score files): host(s) http://127.0.0.1:11503, http://127.0.0.1:11506, http://127.0.0.1:11511 · build(s) 0.33.2-dynres-5-g2b95b4a, 0.33.2-dynres.1-39-g242cad5, 0.33.3-dynres-10-ga523d60 ⚠ MIXED — columns are not one campaign

## T2 — head-to-head, gemma4:31b-nvfp4 think-off: main · 0.33.3 (`mlx0333cu`, Sep 4) · 0.33.2 (`mlx0332cu`, Sep 5). Cross-build by design: the MIXED footer names the three builds; all columns on the CUDA host.

| test | metric | main276_1_gemma4_31b-nvfp4_thinkfalse | mlx0333cu_1_gemma4_31b-nvfp4_thinkfalse | mlx0332cu_1_gemma4_31b-nvfp4_thinkfalse |
|---|---|---|---|---|
| scene | bbox IoU | 0.962 (8192) | 0.962 (8192) | 0.962 (8192) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.751 (8192) | 0.751 (8192) | 0.751 (8192) |
| fine text | 22/16/12/9/7 px | 4/4/4/3/3 (8192) | 4/4/4/3/3 (8192) | 4/4/4/3/3 (8192) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (8192) | error | ✅ ✅ ✅ 5/5 (8192) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| throughput | gen tok/s | 23 | 24 | 15 |
| throughput | prefill tok/s | 681 | 745 | 505 |
| latency | s/req (unique image) | 26.1 | 24.7 | 38.5 |
| latency | req/h (serial) | 138 | 146 | 93 |

Provenance (from score files): host(s) http://127.0.0.1:11503, http://127.0.0.1:11506, http://127.0.0.1:11511 · build(s) 0.33.2-dynres-5-g2b95b4a, 0.33.2-dynres.1-39-g242cad5, 0.33.3-dynres-10-ga523d60 ⚠ MIXED — columns are not one campaign

## T2 — head-to-head, qwen3.8:27b-nvfp4 think-off: main · 0.33.3 (`mlx0333cu`, Sep 4) · 0.33.2 (`mlx0332cu`, Sep 5). Cross-build by design: the MIXED footer names the three builds; all columns on the CUDA host.

| test | metric | main276_1_qwen3_8_27b-nvfp4_thinkfalse | mlx0333cu_1_qwen3_8_27b-nvfp4_thinkfalse | mlx0332cu_1_qwen3_8_27b-nvfp4_thinkfalse |
|---|---|---|---|---|
| scene | bbox IoU | 0.999 (8192) | 0.999 (8192) | 0.999 (8192) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.542 (8192) | 0.542 (8192) | 0.542 (8192) |
| fine text | 22/16/12/9/7 px | 4/4/4/2/0 (8192) | 4/4/4/2/0 (8192) | 4/4/4/2/0 (8192) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| throughput | gen tok/s | 16 | 18 | 13 |
| throughput | prefill tok/s | 5222 | 5430 | 5413 |
| latency | s/req (unique image) | 35.7 | 30.9 | 43.0 |
| latency | req/h (serial) | 101 | 116 | 84 |

Provenance (from score files): host(s) http://127.0.0.1:11503, http://127.0.0.1:11506, http://127.0.0.1:11511 · build(s) 0.33.2-dynres-5-g2b95b4a, 0.33.2-dynres.1-39-g242cad5, 0.33.3-dynres-10-ga523d60 ⚠ MIXED — columns are not one campaign

## T2 — head-to-head, qwen3.6:35b-a3b-nvfp4 think-off: main · 0.33.3 (`mlx0333cu`, Sep 4) · 0.33.2 (`mlx0332cu`, Sep 5). Cross-build by design: the MIXED footer names the three builds; all columns on the CUDA host.

| test | metric | main276_1_qwen3_6_35b-a3b-nvfp4_thinkfalse | mlx0333cu_1_qwen3_6_35b-a3b-nvfp4_thinkfalse | mlx0332cu_1_qwen3_6_35b-a3b-nvfp4_thinkfalse |
|---|---|---|---|---|
| scene | bbox IoU | 0.964 (8192) | 0.964 (8192) | 0.965 (8192) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.506 (8192) | 0.504 (8192) | 0.504 (8192) |
| fine text | 22/16/12/9/7 px | 4/4/4/2/1 (8192) | 4/4/4/2/1 (8192) | 4/4/4/2/1 (8192) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| throughput | gen tok/s | 19 | 20 | 19 |
| throughput | prefill tok/s | 4908 | 5582 | 5374 |
| latency | s/req (unique image) | 28.5 | 27.3 | 29.1 |
| latency | req/h (serial) | 126 | 132 | 124 |

Provenance (from score files): host(s) http://127.0.0.1:11503, http://127.0.0.1:11506, http://127.0.0.1:11511 · build(s) 0.33.2-dynres-5-g2b95b4a, 0.33.2-dynres.1-39-g242cad5, 0.33.3-dynres-10-ga523d60 ⚠ MIXED — columns are not one campaign

## Contract matrix, think-off, campaign `main276_1_`

## Contract matrix (`contract_followed`), think=false

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |
| gemma4:26b-nvfp4 | **MLX** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | 8192 |
| gemma4:31b-nvfp4 | **MLX** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 8192 |
| qwen3.8:27b-nvfp4 | **MLX** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |

`error` = the arm ran and errored (OOM, transport, HTTP 500), so there is no contract to judge. `cap` = generation stopped at the `num_predict` cap rather than finishing, so the cell carries no score (ADR 0012 rule 8). The cap is a separate limit from the `num_ctx` window.

## Contract matrix, think-off, campaign `mlx0333cu_1_`

## Contract matrix (`contract_followed`), think=false

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |
| gemma4:26b-nvfp4 | **MLX** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | 8192 |
| gemma4:31b-nvfp4 | **MLX** | ✅ | ✅ | ✅ | ✅ | error | ✅ | ✅ | error | 8192 |
| qwen3.8:27b-nvfp4 | **MLX** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |

`error` = the arm ran and errored (OOM, transport, HTTP 500), so there is no contract to judge. `cap` = generation stopped at the `num_predict` cap rather than finishing, so the cell carries no score (ADR 0012 rule 8). The cap is a separate limit from the `num_ctx` window.

## Contract matrix, think-off, campaign `mlx0332cu_1_`

## Contract matrix (`contract_followed`), think=false

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |
| gemma4:26b-nvfp4 | **MLX** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | 8192 |
| gemma4:31b-nvfp4 | **MLX** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | 8192 |
| qwen3.8:27b-nvfp4 | **MLX** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |

`error` = the arm ran and errored (OOM, transport, HTTP 500), so there is no contract to judge. `cap` = generation stopped at the `num_predict` cap rather than finishing, so the cell carries no score (ADR 0012 rule 8). The cap is a separate limit from the `num_ctx` window.

## Repeats — the four Sep-4 OOM arms, 3× on main vs 3× on 0.33.3 (Sep-4 discriminator) vs the 0.32.14 baseline (`summarize_reps`)

### gemma4_26b-nvfp4

# no scores files for main x3=main276oom_1_gemma4_26b-nvfp4_thinkfalse,main276oom_2_gemma4_26b-nvfp4_thinkfalse,main276oom_3_gemma4_26b-nvfp4_thinkfalse
# no scores files for 0.33.3 x3=mlx0333cuRR_1_gemma4_26b-nvfp4_thinkfalse,mlx0333cuRR_2_gemma4_26b-nvfp4_thinkfalse,mlx0333cuRR_3_gemma4_26b-nvfp4_thinkfalse
# no scores files for 0.32.14=sync15_1_gemma4_26b-nvfp4_thinkfalse
| metric |  |
|---|
| **num_ctx rung** |  |
| **num_predict** |  |

Within-arm spread (max-min), the bar any cross-arm claim must clear:

### gemma4_31b-nvfp4

# no scores files for main x3=main276oom_1_gemma4_31b-nvfp4_thinkfalse,main276oom_2_gemma4_31b-nvfp4_thinkfalse,main276oom_3_gemma4_31b-nvfp4_thinkfalse
# no scores files for 0.33.3 x3=mlx0333cuRR_1_gemma4_31b-nvfp4_thinkfalse,mlx0333cuRR_2_gemma4_31b-nvfp4_thinkfalse,mlx0333cuRR_3_gemma4_31b-nvfp4_thinkfalse
# no scores files for 0.32.14=sync15_1_gemma4_31b-nvfp4_thinkfalse
| metric |  |
|---|
| **num_ctx rung** |  |
| **num_predict** |  |

Within-arm spread (max-min), the bar any cross-arm claim must clear:


**Derived, not generator output** (read from `done_reason`, SPEC H4b; the score
files are the veto): 135 of 135 arms converged across the five cells, all at
rung 8192.

## Step 3a — preflight on the `main` binary

Profile `cuda-dynres-903`, `--quality`, canary on `:11437` mirroring the deployed
recipe: **PASS 20 / SKIP 8 / 0 fail** (`preflight/runs/full-main-a523d60b.json`).
Compared check by check with the tagged run `full-0333-tagged.json`: 28 checks,
**0 differ** beyond the image tag, the version stamp and think-format token
counts; `payload_pin` `0f3a71be1`, poison probe clean over 33 nodes. The GGML path
is untouched by what landed on `main`.

## Step 3b — the automatic-context clamp on a real load

Unpinned requests (server default `num_ctx` 262144) on a 16 GiB-headroom
container, from the runner's admission lines:

```
"MLX context clamped to fit VRAM" model=gemma4:31b-nvfp4 requested=262144 using=65536 weights="17.3 GiB" kv="5.8 GiB" headroom="14.5 GiB" budget="39.9 GiB"
"MLX admission priced the context rung" model=gemma4:31b-nvfp4 num_ctx=65536 num_ctx_auto=true weights="17.3 GiB" kv="5.8 GiB" headroom="14.5 GiB" need="37.6 GiB" budget="39.9 GiB"
"MLX admission priced the context rung" model=gemma4:12b-nvfp4 num_ctx=262144 num_ctx_auto=true weights="7.1 GiB" kv="4.3 GiB" headroom="14.5 GiB" need="25.9 GiB" budget="39.9 GiB"
```

31b's automatic rung clamps 262144 → 65536 and is served; 12b's fits at 262144
(need 25.9 GiB) and is not clamped — `/api/ps` reports `context_length=262144`.
(The `/api/ps` reading for 31b was lost to a quoting bug in the probe; the runner's
clamp line above is the primary evidence.)

## #279 — a rung change reloads an MLX runner, live proof

One container, 35 GiB cap, gemma4:31b; `runner starts` counted from the server log:

```
##### FIX: maxusai/ollama:reloadfix -> {"version":"0.33.3-dynres-11-gd9055b7"} 15:53:16
  1. 31b@8192  -> 200 452s ok 
  2. 31b@65536 -> 500 4s model requires 37.6 GiB but OLLAMA_MLX_MEMORY_LIMIT caps the MLX budget at 35.0 GiB | runner starts +0
  3. 31b@8192  -> 200 53s ok  | runner starts +1
  4. 31b@4096  -> 200 32s ok  | runner starts +1
  5. 31b@8192  -> 200 32s ok  | runner starts +1
  6. 31b@8192  -> 200 0s ok  | runner starts +0 (want +0: warm)
##### CONTROL (main, no fix): maxusai/ollama:main-a523d60b -> {"version":"0.33.3-dynres-10-ga523d60"} 16:03:07
  1. 31b@8192  -> 200 251s ok 
  2. 31b@65536 -> 200 1s ok | runner starts +0
  3. 31b@8192  -> 200 1s ok  | runner starts +0
  4. 31b@4096  -> 200 1s ok  | runner starts +0
  5. 31b@8192  -> 200 1s ok  | runner starts +0
  6. 31b@8192  -> 200 1s ok  | runner starts +0 (want +0: warm)
```

FIX (`0.33.3-dynres-11-gd9055b7`): 65536 after a warm 8192 is **refused at
admission**; 8192 and 4096 each reload (a smaller rung releases the bigger
window); the same rung twice is served warm. CONTROL (`main-a523d60b`): every rung
change served warm in ~1 s with zero runner starts — the gap as found.

## qwen3.6 `bbox_contract_reasoning` — one discrete flip, repeated n=3

`contract_followed` read ❌ on `main` and on 0.33.2 (both stopped at 370 tokens) and
✅ on 0.33.3 (419 tokens). One arm on the loop-prone model is not a verdict
(ADR 0029, marginal-probes rule); the arm was repeated three times on each build:

Contract matrix per repeat (generator output; `bcreasoning` is the column in question):

| run | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| main rep 1 | **MLX** | — | — | ❌ | — | — | — | — | — | 8192 |
| main rep 2 | **MLX** | — | — | ❌ | — | — | — | — | — | 8192 |
| main rep 3 | **MLX** | — | — | ✅ | — | — | — | — | — | 8192 |
| 0.33.3 (tagged image) rep 1 | **MLX** | — | — | ❌ | — | — | — | — | — | 8192 |
| 0.33.3 (tagged image) rep 2 | **MLX** | — | — | ❌ | — | — | — | — | — | 8192 |
| 0.33.3 (tagged image) rep 3 | **MLX** | — | — | ✅ | — | — | — | — | — | 8192 |

**Derived, not generator output** (`eval_count` and `contract_followed` read from the six score files; they are the veto):

- main: ❌ at 370 tokens · ❌ at 370 tokens · ✅ at 395 tokens
- 0.33.3 (tagged image): ❌ at 370 tokens · ❌ at 370 tokens · ✅ at 419 tokens

The same arm flips between repeats of one build, so the single-arm difference seen in the
campaign tables is run-to-run variance of greedy decoding on this model — the ADR 0029
class — not a build effect. It stays out of the verdict.

