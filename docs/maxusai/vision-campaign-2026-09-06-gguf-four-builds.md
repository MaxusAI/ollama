# Vision validation 2026-09-06: the GGUF (llama-server) path across 0.32.14 · 0.33.2 · 0.33.3 · main

> Verdict: **no regression on the GGUF path, think off.** `main` (`a523d60b`) equals the
> tagged 0.33.3 on all 56 quality cells and all 64 contract cells. 0.33.3 differs from
> 0.33.2 on 7 quality cells — five IoU moves of ≤ 0.009 and two single-observation flips
> listed below — and on 4 contract cells, in both directions, on the arms that flip between
> repeats within one build. 24/24 suites converged at 8192 with 0 OOMs and 0 errors on every
> build. The throughput columns were measured beside production and are not comparable.
> Think-on rows follow (overnight 2026-09-06/07, section at the end).

## Provenance

- **Date / host:** 2026-09-06 17:46–21:25 local, CUDA box (RTX PRO 6000 Blackwell,
  97.9 GB), shared with the production endpoint on `:11497` (`0.33.2-dynres-5-g2b95b4a`,
  teacher-v3 fetch loop) throughout.
- **Builds under test**, one container each on `:11516`, run back to back:

  | campaign prefix | image | version stamp | ran | wall |
  |---|---|---|---|---|
  | `ggml0332_1_1_` | `maxusai/ollama:sync-0.33.2` | `0.33.2-dynres-5-g2b95b4a` | 17:46–19:02 | 76 min |
  | `ggml0333_1_1_` | `maxusai/ollama:sync-0.33.3` | `0.33.3-dynres-0-g0c4f09d` | 19:03–20:13 | 70 min |
  | `ggmlmain_1_1_` | `maxusai/ollama:main-a523d60b` | `0.33.3-dynres-10-ga523d60` | 20:13–21:25 | 72 min |

  `sync-0.33.3` and `main-a523d60b` share the native payload (llama.cpp `b10760`
  `0f3a71be1`, patchset 903 + fork band); `main` differs by its Go binary only (#271, #276,
  #277, #278, #279, #280, #281 — none on the llama-server path). The campaign prefix carries
  the driver's rep index: `TAG_PREFIX=ggml0332_1_` writes tags
  `ggml0332_1_1_<model>_thinkfalse`, so the generators take `--prefix ggml0332_1_1_`.
- **Reference:** `sync15_1_` (2026-08-24, `0.32.14-dynres-108-g76918a7`, `:11502`, quiet
  GPU) — the last GGUF campaign before the 0.33 line
  ([write-up](vision-campaign-2026-08-24-sync15nt-thinkon.md)). Its ladder converged at
  16384 for 31b, qwen3.6 and nemotron q4; the T2 cells carry the rung in parentheses.
- **Models:** eight GGUF quantisations — gemma4 `31b-it`, `26b-a4b-it`, `e4b-it`, `e2b-it`
  (all q4_K_M), `qwen3.8:27b-q4_K_M`, `qwen3.6:35b-a3b-q4_K_M`, `nemotron3:33b-q4_K_M`,
  `nemotron3:33b-q8`.
- **Endpoint / sampling:** `run_engine_compare.sh` defaults — `/api/chat`, think off at
  temperature 0 (ADR 0029), `num_predict` 2200, ladder from `num_ctx` 8192 to 65536, one
  runner, cold restart per cell, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_GPU_OVERHEAD` 16 GiB
  (held constant across the three builds; the llama-server path prices `n_ctx` itself).
- **Scripts:** `claude-scratch/gate-ggml.sh <false|on> <image> <prefix>` per build, sequenced
  by `run-phase31.sh`; render `render-ggml.sh false` → every table below is generator output
  (ADR 0012 rule 1: T1 `summarize_engine_compare.py`, T2 `summarize_head_to_head.py`,
  `summarize_contract_matrix.py`), rendered once all three campaigns had completed (rule 8).
  Driver logs `preflight-runs/ggml0332_1_thinkfalse.log`, `ggml0333_1_thinkfalse.log`,
  `ggmlmain_1_thinkfalse.log`; the render `preflight-runs/ggml-render-thinkfalse.md`.

## Driver summaries

| build | OOMs | errors | suites | not converged |
|---|---|---|---|---|
| 0.33.2 | 0 | 0 | 8 | 0 |
| 0.33.3 | 0 | 0 | 8 | 0 |
| main | 0 | 0 | 8 | 0 |

Every cell on every build converged at the first rung, 8192.

## What differs between the builds

Derived from the eight T2 tables below by column diff over the quality rows (7 rows × 8
models = 56 cells; throughput and latency excluded, see the next section) and from the four
contract matrices (8 arms × 8 models = 64 cells). **Veto:** the T2 tables and contract
matrices below are the primary record; any cell there that contradicts this section wins.

- **0.33.3 vs main: 0/56 quality cells, 0/64 contract cells differ.** Same native payload,
  Go-side changes only, none on this path.
- **0.33.2 vs 0.33.3: 7/56 quality cells differ:**

  | model | test · metric | 0.33.2 | 0.33.3 = main | 0.32.14 |
  |---|---|---|---|---|
  | gemma4:26b-a4b | scene · bbox IoU | 0.975 | 0.972 | 0.973 |
  | gemma4:26b-a4b | document · name_bbox IoU | 0.753 | 0.754 | 0.756 |
  | gemma4:e4b | multi (3 img) · q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 | ✅ ❌ ✅ 4/5 | ✅ ✅ ✅ 5/5 |
  | qwen3.6:35b-a3b | scene · bbox IoU | 0.976 | 0.974 | 0.975 |
  | qwen3.6:35b-a3b | document · name_bbox IoU | 0.607 | 0.616 | 0.607 |
  | nemotron3:33b-q4 | scene · bbox IoU | 0.863 | 0.862 | 0.870 |
  | nemotron3:33b-q8 | document · name_bbox IoU | 0.162 | 0.044 | 0.044 |

  Five are IoU moves of at most 0.009. Two are single observations: e4b's multi-image `q2`
  (0.33.2 and 0.32.14 pass it, 0.33.3 and `main` fail it — `q2_right` in the T1 cell) and
  nemotron q8's `name_bbox` (0.33.2 is the outlier against the other three). Neither is a
  regression call at n=1 — the rule since the qwen3.6 retraction is n≥5 with a control in the
  same container; that repeat is a follow-up, not run here.
- **0.33.2 vs 0.33.3 contract matrix: 4/64 differ** — 26b `bcreasoning` ❌→✅ and
  `bcperobject` ✅→❌, nemotron q4 `bcreasoning` ✅→❌, nemotron q8 `bcmulti` ❌→✅. Both
  directions, on the arms that flip between repeats within one build (the qwen3.6 reasoning
  arm did ❌/❌/✅ on both builds in the `main` validation).
- **0.32.14 vs 0.33.3: 15/56 quality cells differ.** Eight are IoU moves of ≤ 0.026 (four
  of them across the rung change — 31b, qwen3.6 and nemotron q4 converged at 16384 there;
  the largest is e2b's scene 0.087 → 0.061, on a model that cannot ground). The other seven:
  gemma4:31b `name_bbox` 0.752 (16384) → 0.708 (8192), also across the rung change; e4b
  fine text 12 px 0 → 3 and anchored chart 4/5 → 5/5, against its multi `q2` flip above;
  e2b fine text `capped (131072)` → `0/0/0/0/0` converged at 8192, and its multi `q4-bbox`
  ❌ → ✅ on both multi rows. Contract matrix 6/64, same arms as above (26b `bcreasoning` /
  `bcpinned` / `bcadvreal` ❌→✅, e4b `bcmulti` ✅→❌, nemotron q4 and q8 `bcreasoning`
  ✅→❌).
- Everything else — scene labels / serial, invoice extraction, fine-text tiers, anchored
  multi-image — is identical across all four builds.

## Throughput and latency are not comparable in this render

All three 2026-09-06 campaigns ran beside the production teacher-v3 loop on the same GPU;
the 0.32.14 reference ran on a quiet GPU. gemma4:31b gen tok/s reads 52 there against
20 / 21 / 25 today, and today's spread on nemotron q4 (75 / 131 / 36) and e4b (34 / 35 / 104)
is contention, not builds. The T1 and T2 throughput columns below are rendered because the
generators render them; do not read build differences from them. A controlled throughput
comparison needs the quiet GPU and is not queued.

## Think-on

Row 2 of the plan — GGUF think-on on the same three builds (`ggml0332nt_1_1_`,
`ggml0333nt_1_1_`, `ggmlmainnt_1_1_`, ladder from 16384) — runs overnight 2026-09-06/07
(`overnight-chain.sh`); its render (`render-ggml.sh on`) is appended here when complete.

## Tables — generator output, `render-ggml.sh false`, rendered 2026-09-06 21:26

## T1 — campaign `sync15_1_`

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 0.962 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 8192 | 0.973 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:e4b-it-q4_K_M | GGUF | 8192 | 0.341 | 3/7 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ❌ | 3 |
| gemma4:e2b-it-q4_K_M | GGUF | 8192/131072 ⚠ | 0.087 | 1/8 · 6/6 · 6/6 | ✅ | 1/5 · 1/5 · ✅ | 0 |
| qwen3.8:27b-q4_K_M | GGUF | 8192 | 0.977 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 16384 | 0.975 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 0.870 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q8 | GGUF | 8192 | 0.870 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 4 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 538 | 52 | 436 | 14.2 | 253 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 536 | 181 | 1327 | 4.2 | 852 |
| gemma4:e4b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 0 | 0 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 501 | 180 | 3054 | 3.3 | 1080 |
| gemma4:e2b-it-q4_K_M | GGUF | 8192/131072 ⚠ | capped | capped | capped | capped | capped | ❌ q4_bbox_hit | ❌ q4_bbox_hit | — | 697 | 235 | 3446 | 3.5 | 1042 |
| qwen3.8:27b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 2 | 1 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 544 | 70 | 1815 | 9.2 | 391 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 2 | 2 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 548 | 108 | 3323 | 5.8 | 616 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 270 | 5038 | 2.4 | 1485 |
| nemotron3:33b-q8 | GGUF | 8192 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 227 | 4956 | 2.8 | 1288 |

Provenance (from score files): host(s) http://127.0.0.1:11502 · build(s) 0.32.14-dynres-108-g76918a7 · think=false

## T1 — campaign `ggml0332_1_1_`

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | 8192 | 0.965 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 8192 | 0.975 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:e4b-it-q4_K_M | GGUF | 8192 | 0.354 | 3/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ❌ | 2 |
| gemma4:e2b-it-q4_K_M | GGUF | 8192 | 0.061 | 1/7 · 6/6 · 6/6 | ✅ | 1/5 · 1/5 · ✅ | 0 |
| qwen3.8:27b-q4_K_M | GGUF | 8192 | 0.977 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 8192 | 0.976 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q4_K_M | GGUF | 8192 | 0.863 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q8 | GGUF | 8192 | 0.870 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 4 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 538 | 20 | 436 | 30.8 | 117 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 536 | 37 | 532 | 17.6 | 204 |
| gemma4:e4b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 3 | 0 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 443 | 34 | 5410 | 13.4 | 269 |
| gemma4:e2b-it-q4_K_M | GGUF | 8192 | 0 | 0 | 0 | 0 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 630 | 67 | 2269 | 10.1 | 356 |
| qwen3.8:27b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 2 | 1 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 544 | 25 | 1094 | 24.0 | 150 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 2 | 2 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 550 | 34 | 1821 | 17.4 | 206 |
| nemotron3:33b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 75 | 2509 | 7.9 | 455 |
| nemotron3:33b-q8 | GGUF | 8192 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 55 | 2897 | 10.2 | 352 |

Provenance (from score files): host(s) http://127.0.0.1:11516 · build(s) 0.33.2-dynres-5-g2b95b4a · think=false

## T1 — campaign `ggml0333_1_1_`

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | 8192 | 0.965 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 8192 | 0.972 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:e4b-it-q4_K_M | GGUF | 8192 | 0.354 | 3/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ❌ | 2 |
| gemma4:e2b-it-q4_K_M | GGUF | 8192 | 0.061 | 1/7 · 6/6 · 6/6 | ✅ | 1/5 · 1/5 · ✅ | 0 |
| qwen3.8:27b-q4_K_M | GGUF | 8192 | 0.977 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 8192 | 0.974 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q4_K_M | GGUF | 8192 | 0.862 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q8 | GGUF | 8192 | 0.870 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 4 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 538 | 21 | 452 | 29.5 | 122 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 536 | 36 | 819 | 17.1 | 210 |
| gemma4:e4b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 3 | 0 | 0 | ❌ q2_right | ✅ q1 + q2 + q4-bbox | — | 443 | 35 | 6735 | 13.0 | 278 |
| gemma4:e2b-it-q4_K_M | GGUF | 8192 | 0 | 0 | 0 | 0 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 630 | 41 | 2166 | 16.1 | 223 |
| qwen3.8:27b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 2 | 1 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 544 | 30 | 1066 | 20.4 | 177 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 2 | 2 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 544 | 36 | 2420 | 16.3 | 220 |
| nemotron3:33b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 131 | 3781 | 4.6 | 779 |
| nemotron3:33b-q8 | GGUF | 8192 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 100 | 3813 | 5.8 | 616 |

Provenance (from score files): host(s) http://127.0.0.1:11516 · build(s) 0.33.3-dynres-0-g0c4f09d · think=false

## T1 — campaign `ggmlmain_1_1_`

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | 8192 | 0.965 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 8192 | 0.972 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:e4b-it-q4_K_M | GGUF | 8192 | 0.354 | 3/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ❌ | 2 |
| gemma4:e2b-it-q4_K_M | GGUF | 8192 | 0.061 | 1/7 · 6/6 · 6/6 | ✅ | 1/5 · 1/5 · ✅ | 0 |
| qwen3.8:27b-q4_K_M | GGUF | 8192 | 0.977 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 8192 | 0.974 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q4_K_M | GGUF | 8192 | 0.862 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q8 | GGUF | 8192 | 0.870 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 4 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 538 | 25 | 452 | 25.2 | 143 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 536 | 35 | 760 | 17.7 | 204 |
| gemma4:e4b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 3 | 0 | 0 | ❌ q2_right | ✅ q1 + q2 + q4-bbox | — | 443 | 104 | 8324 | 4.5 | 804 |
| gemma4:e2b-it-q4_K_M | GGUF | 8192 | 0 | 0 | 0 | 0 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 630 | 39 | 2169 | 17.0 | 212 |
| qwen3.8:27b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 2 | 1 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 544 | 42 | 1509 | 14.6 | 247 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 2 | 2 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 544 | 32 | 2009 | 18.2 | 198 |
| nemotron3:33b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 36 | 1518 | 16.0 | 225 |
| nemotron3:33b-q8 | GGUF | 8192 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 29 | 940 | 20.5 | 175 |

Provenance (from score files): host(s) http://127.0.0.1:11516 · build(s) 0.33.3-dynres-10-ga523d60 · think=false

## T2 — gemma4:31b-it-q4_K_M think=false: 0.32.14 · 0.33.2 · 0.33.3 · main (cross-build by design; MIXED footer names the builds; all columns on the CUDA host)

| test | metric | sync15_1_gemma4_31b-it-q4_K_M_thinkfalse | ggml0332_1_1_gemma4_31b-it-q4_K_M_thinkfalse | ggml0333_1_1_gemma4_31b-it-q4_K_M_thinkfalse | ggmlmain_1_1_gemma4_31b-it-q4_K_M_thinkfalse |
|---|---|---|---|---|---|
| scene | bbox IoU | 0.962 (16384) | 0.965 (8192) | 0.965 (8192) | 0.965 (8192) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.752 (16384) | 0.708 (8192) | 0.708 (8192) | 0.708 (8192) |
| fine text | 22/16/12/9/7 px | 4/4/4/4/3 (16384) | 4/4/4/4/3 (8192) | 4/4/4/4/3 (8192) | 4/4/4/4/3 (8192) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| throughput | gen tok/s | 52 | 20 | 21 | 25 |
| throughput | prefill tok/s | 436 | 436 | 452 | 452 |
| latency | s/req (unique image) | 14.2 | 30.8 | 29.5 | 25.2 |
| latency | req/h (serial) | 253 | 117 | 122 | 143 |

Provenance (from score files): host(s) http://127.0.0.1:11502, http://127.0.0.1:11516 · build(s) 0.32.14-dynres-108-g76918a7, 0.33.2-dynres-5-g2b95b4a, 0.33.3-dynres-0-g0c4f09d, 0.33.3-dynres-10-ga523d60 ⚠ MIXED — columns are not one campaign

## T2 — gemma4:26b-a4b-it-q4_K_M think=false: 0.32.14 · 0.33.2 · 0.33.3 · main (cross-build by design; MIXED footer names the builds; all columns on the CUDA host)

| test | metric | sync15_1_gemma4_26b-a4b-it-q4_K_M_thinkfalse | ggml0332_1_1_gemma4_26b-a4b-it-q4_K_M_thinkfalse | ggml0333_1_1_gemma4_26b-a4b-it-q4_K_M_thinkfalse | ggmlmain_1_1_gemma4_26b-a4b-it-q4_K_M_thinkfalse |
|---|---|---|---|---|---|
| scene | bbox IoU | 0.973 (8192) | 0.975 (8192) | 0.972 (8192) | 0.972 (8192) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.756 (8192) | 0.753 (8192) | 0.754 (8192) | 0.754 (8192) |
| fine text | 22/16/12/9/7 px | 4/4/4/3/3 (8192) | 4/4/4/3/3 (8192) | 4/4/4/3/3 (8192) | 4/4/4/3/3 (8192) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| throughput | gen tok/s | 181 | 37 | 36 | 35 |
| throughput | prefill tok/s | 1327 | 532 | 819 | 760 |
| latency | s/req (unique image) | 4.2 | 17.6 | 17.1 | 17.7 |
| latency | req/h (serial) | 852 | 204 | 210 | 204 |

Provenance (from score files): host(s) http://127.0.0.1:11502, http://127.0.0.1:11516 · build(s) 0.32.14-dynres-108-g76918a7, 0.33.2-dynres-5-g2b95b4a, 0.33.3-dynres-0-g0c4f09d, 0.33.3-dynres-10-ga523d60 ⚠ MIXED — columns are not one campaign

## T2 — gemma4:e4b-it-q4_K_M think=false: 0.32.14 · 0.33.2 · 0.33.3 · main (cross-build by design; MIXED footer names the builds; all columns on the CUDA host)

| test | metric | sync15_1_gemma4_e4b-it-q4_K_M_thinkfalse | ggml0332_1_1_gemma4_e4b-it-q4_K_M_thinkfalse | ggml0333_1_1_gemma4_e4b-it-q4_K_M_thinkfalse | ggmlmain_1_1_gemma4_e4b-it-q4_K_M_thinkfalse |
|---|---|---|---|---|---|
| scene | bbox IoU | 0.341 (8192) | 0.354 (8192) | 0.354 (8192) | 0.354 (8192) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ❌, ✅ | 5/5, 5/5, ❌, ✅ | 5/5, 5/5, ❌, ✅ | 5/5, 5/5, ❌, ✅ |
| document | name_bbox IoU | 0.000 (8192) | 0.000 (8192) | 0.000 (8192) | 0.000 (8192) |
| fine text | 22/16/12/9/7 px | 4/4/0/0/0 (8192) | 4/4/3/0/0 (8192) | 4/4/3/0/0 (8192) | 4/4/3/0/0 (8192) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ❌ ✅ 4/5 (8192) | ✅ ❌ ✅ 4/5 (8192) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 4/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| throughput | gen tok/s | 180 | 34 | 35 | 104 |
| throughput | prefill tok/s | 3054 | 5410 | 6735 | 8324 |
| latency | s/req (unique image) | 3.3 | 13.4 | 13.0 | 4.5 |
| latency | req/h (serial) | 1080 | 269 | 278 | 804 |

Provenance (from score files): host(s) http://127.0.0.1:11502, http://127.0.0.1:11516 · build(s) 0.32.14-dynres-108-g76918a7, 0.33.2-dynres-5-g2b95b4a, 0.33.3-dynres-0-g0c4f09d, 0.33.3-dynres-10-ga523d60 ⚠ MIXED — columns are not one campaign

## T2 — gemma4:e2b-it-q4_K_M think=false: 0.32.14 · 0.33.2 · 0.33.3 · main (cross-build by design; MIXED footer names the builds; all columns on the CUDA host)

| test | metric | sync15_1_gemma4_e2b-it-q4_K_M_thinkfalse | ggml0332_1_1_gemma4_e2b-it-q4_K_M_thinkfalse | ggml0333_1_1_gemma4_e2b-it-q4_K_M_thinkfalse | ggmlmain_1_1_gemma4_e2b-it-q4_K_M_thinkfalse |
|---|---|---|---|---|---|
| scene | bbox IoU | 0.087 (8192) | 0.061 (8192) | 0.061 (8192) | 0.061 (8192) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 1/5, 1/5, ✅, ✅ | 1/5, 1/5, ✅, ✅ | 1/5, 1/5, ✅, ✅ | 1/5, 1/5, ✅, ✅ |
| document | name_bbox IoU | 0.000 (8192) | 0.000 (8192) | 0.000 (8192) | 0.000 (8192) |
| fine text | 22/16/12/9/7 px | capped (131072) | 0/0/0/0/0 (8192) | 0/0/0/0/0 (8192) | 0/0/0/0/0 (8192) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ❌ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ❌ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| throughput | gen tok/s | 235 | 67 | 41 | 39 |
| throughput | prefill tok/s | 3446 | 2269 | 2166 | 2169 |
| latency | s/req (unique image) | 3.5 | 10.1 | 16.1 | 17.0 |
| latency | req/h (serial) | 1042 | 356 | 223 | 212 |

Provenance (from score files): host(s) http://127.0.0.1:11502, http://127.0.0.1:11516 · build(s) 0.32.14-dynres-108-g76918a7, 0.33.2-dynres-5-g2b95b4a, 0.33.3-dynres-0-g0c4f09d, 0.33.3-dynres-10-ga523d60 ⚠ MIXED — columns are not one campaign

## T2 — qwen3.8:27b-q4_K_M think=false: 0.32.14 · 0.33.2 · 0.33.3 · main (cross-build by design; MIXED footer names the builds; all columns on the CUDA host)

| test | metric | sync15_1_qwen3_8_27b-q4_K_M_thinkfalse | ggml0332_1_1_qwen3_8_27b-q4_K_M_thinkfalse | ggml0333_1_1_qwen3_8_27b-q4_K_M_thinkfalse | ggmlmain_1_1_qwen3_8_27b-q4_K_M_thinkfalse |
|---|---|---|---|---|---|
| scene | bbox IoU | 0.977 (8192) | 0.977 (8192) | 0.977 (8192) | 0.977 (8192) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.550 (8192) | 0.550 (8192) | 0.550 (8192) | 0.550 (8192) |
| fine text | 22/16/12/9/7 px | 4/4/4/2/1 (8192) | 4/4/4/2/1 (8192) | 4/4/4/2/1 (8192) | 4/4/4/2/1 (8192) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ❌ 5/5 (8192) | ✅ ✅ ❌ 5/5 (8192) | ✅ ✅ ❌ 5/5 (8192) | ✅ ✅ ❌ 5/5 (8192) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| throughput | gen tok/s | 70 | 25 | 30 | 42 |
| throughput | prefill tok/s | 1815 | 1094 | 1066 | 1509 |
| latency | s/req (unique image) | 9.2 | 24.0 | 20.4 | 14.6 |
| latency | req/h (serial) | 391 | 150 | 177 | 247 |

Provenance (from score files): host(s) http://127.0.0.1:11502, http://127.0.0.1:11516 · build(s) 0.32.14-dynres-108-g76918a7, 0.33.2-dynres-5-g2b95b4a, 0.33.3-dynres-0-g0c4f09d, 0.33.3-dynres-10-ga523d60 ⚠ MIXED — columns are not one campaign

## T2 — qwen3.6:35b-a3b-q4_K_M think=false: 0.32.14 · 0.33.2 · 0.33.3 · main (cross-build by design; MIXED footer names the builds; all columns on the CUDA host)

| test | metric | sync15_1_qwen3_6_35b-a3b-q4_K_M_thinkfalse | ggml0332_1_1_qwen3_6_35b-a3b-q4_K_M_thinkfalse | ggml0333_1_1_qwen3_6_35b-a3b-q4_K_M_thinkfalse | ggmlmain_1_1_qwen3_6_35b-a3b-q4_K_M_thinkfalse |
|---|---|---|---|---|---|
| scene | bbox IoU | 0.975 (16384) | 0.976 (8192) | 0.974 (8192) | 0.974 (8192) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.607 (16384) | 0.607 (8192) | 0.616 (8192) | 0.616 (8192) |
| fine text | 22/16/12/9/7 px | 4/4/4/2/2 (16384) | 4/4/4/2/2 (8192) | 4/4/4/2/2 (8192) | 4/4/4/2/2 (8192) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| throughput | gen tok/s | 108 | 34 | 36 | 32 |
| throughput | prefill tok/s | 3323 | 1821 | 2420 | 2009 |
| latency | s/req (unique image) | 5.8 | 17.4 | 16.3 | 18.2 |
| latency | req/h (serial) | 616 | 206 | 220 | 198 |

Provenance (from score files): host(s) http://127.0.0.1:11502, http://127.0.0.1:11516 · build(s) 0.32.14-dynres-108-g76918a7, 0.33.2-dynres-5-g2b95b4a, 0.33.3-dynres-0-g0c4f09d, 0.33.3-dynres-10-ga523d60 ⚠ MIXED — columns are not one campaign

## T2 — nemotron3:33b-q4_K_M think=false: 0.32.14 · 0.33.2 · 0.33.3 · main (cross-build by design; MIXED footer names the builds; all columns on the CUDA host)

| test | metric | sync15_1_nemotron3_33b-q4_K_M_thinkfalse | ggml0332_1_1_nemotron3_33b-q4_K_M_thinkfalse | ggml0333_1_1_nemotron3_33b-q4_K_M_thinkfalse | ggmlmain_1_1_nemotron3_33b-q4_K_M_thinkfalse |
|---|---|---|---|---|---|
| scene | bbox IoU | 0.870 (16384) | 0.863 (8192) | 0.862 (8192) | 0.862 (8192) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.044 (16384) | 0.044 (8192) | 0.044 (8192) | 0.044 (8192) |
| fine text | 22/16/12/9/7 px | 4/4/4/3/0 (16384) | 4/4/4/3/0 (8192) | 4/4/4/3/0 (8192) | 4/4/4/3/0 (8192) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| throughput | gen tok/s | 270 | 75 | 131 | 36 |
| throughput | prefill tok/s | 5038 | 2509 | 3781 | 1518 |
| latency | s/req (unique image) | 2.4 | 7.9 | 4.6 | 16.0 |
| latency | req/h (serial) | 1485 | 455 | 779 | 225 |

Provenance (from score files): host(s) http://127.0.0.1:11502, http://127.0.0.1:11516 · build(s) 0.32.14-dynres-108-g76918a7, 0.33.2-dynres-5-g2b95b4a, 0.33.3-dynres-0-g0c4f09d, 0.33.3-dynres-10-ga523d60 ⚠ MIXED — columns are not one campaign

## T2 — nemotron3:33b-q8 think=false: 0.32.14 · 0.33.2 · 0.33.3 · main (cross-build by design; MIXED footer names the builds; all columns on the CUDA host)

| test | metric | sync15_1_nemotron3_33b-q8_thinkfalse | ggml0332_1_1_nemotron3_33b-q8_thinkfalse | ggml0333_1_1_nemotron3_33b-q8_thinkfalse | ggmlmain_1_1_nemotron3_33b-q8_thinkfalse |
|---|---|---|---|---|---|
| scene | bbox IoU | 0.870 (8192) | 0.870 (8192) | 0.870 (8192) | 0.870 (8192) |
| scene | labels / serial | 6/6, ❌ | 6/6, ❌ | 6/6, ❌ | 6/6, ❌ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.044 (8192) | 0.162 (8192) | 0.044 (8192) | 0.044 (8192) |
| fine text | 22/16/12/9/7 px | 4/4/4/3/0 (8192) | 4/4/4/3/0 (8192) | 4/4/4/3/0 (8192) | 4/4/4/3/0 (8192) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) | ✅ ✅ ✅ 5/5 (8192) |
| throughput | gen tok/s | 227 | 55 | 100 | 29 |
| throughput | prefill tok/s | 4956 | 2897 | 3813 | 940 |
| latency | s/req (unique image) | 2.8 | 10.2 | 5.8 | 20.5 |
| latency | req/h (serial) | 1288 | 352 | 616 | 175 |

Provenance (from score files): host(s) http://127.0.0.1:11502, http://127.0.0.1:11516 · build(s) 0.32.14-dynres-108-g76918a7, 0.33.2-dynres-5-g2b95b4a, 0.33.3-dynres-0-g0c4f09d, 0.33.3-dynres-10-ga523d60 ⚠ MIXED — columns are not one campaign

## Contract matrix, think=false, campaign `sync15_1_`

## Contract matrix (`contract_followed`), think=false

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 16384 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ | 8192 |
| gemma4:e4b-it-q4_K_M | GGUF | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | 8192 |
| gemma4:e2b-it-q4_K_M | GGUF | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 8192 |
| qwen3.8:27b-q4_K_M | GGUF | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 8192 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 16384 |
| nemotron3:33b-q4_K_M | GGUF | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 16384 |
| nemotron3:33b-q8 | GGUF | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |

`error` = the arm ran and errored (OOM, transport, HTTP 500), so there is no contract to judge. `cap` = generation stopped at the `num_predict` cap rather than finishing, so the cell carries no score (ADR 0012 rule 8). The cap is a separate limit from the `num_ctx` window.

## Contract matrix, think=false, campaign `ggml0332_1_1_`

## Contract matrix (`contract_followed`), think=false

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 8192 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | 8192 |
| gemma4:e4b-it-q4_K_M | GGUF | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | 8192 |
| gemma4:e2b-it-q4_K_M | GGUF | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 8192 |
| qwen3.8:27b-q4_K_M | GGUF | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 8192 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |
| nemotron3:33b-q4_K_M | GGUF | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |
| nemotron3:33b-q8 | GGUF | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |

`error` = the arm ran and errored (OOM, transport, HTTP 500), so there is no contract to judge. `cap` = generation stopped at the `num_predict` cap rather than finishing, so the cell carries no score (ADR 0012 rule 8). The cap is a separate limit from the `num_ctx` window.

## Contract matrix, think=false, campaign `ggml0333_1_1_`

## Contract matrix (`contract_followed`), think=false

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 8192 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | 8192 |
| gemma4:e4b-it-q4_K_M | GGUF | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | 8192 |
| gemma4:e2b-it-q4_K_M | GGUF | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 8192 |
| qwen3.8:27b-q4_K_M | GGUF | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 8192 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |
| nemotron3:33b-q4_K_M | GGUF | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |
| nemotron3:33b-q8 | GGUF | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |

`error` = the arm ran and errored (OOM, transport, HTTP 500), so there is no contract to judge. `cap` = generation stopped at the `num_predict` cap rather than finishing, so the cell carries no score (ADR 0012 rule 8). The cap is a separate limit from the `num_ctx` window.

## Contract matrix, think=false, campaign `ggmlmain_1_1_`

## Contract matrix (`contract_followed`), think=false

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | 8192 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | 8192 |
| gemma4:e4b-it-q4_K_M | GGUF | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | 8192 |
| gemma4:e2b-it-q4_K_M | GGUF | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 8192 |
| qwen3.8:27b-q4_K_M | GGUF | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 8192 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |
| nemotron3:33b-q4_K_M | GGUF | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |
| nemotron3:33b-q8 | GGUF | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |

`error` = the arm ran and errored (OOM, transport, HTTP 500), so there is no contract to judge. `cap` = generation stopped at the `num_predict` cap rather than finishing, so the cell carries no score (ADR 0012 rule 8). The cap is a separate limit from the `num_ctx` window.

