# Vision validation 2026-09-13: Qwen2.5-VL 3B · 7B · 32B at q4_K_M and q8_0 on the GGUF path

> Verdict: **q8_0 buys nothing measurable over q4_K_M at any size, and costs 31–43 % of
> generation speed.** At 32B every quality cell matches except document `name_bbox` IoU
> (0.780 against 0.770); at 3B and 7B the few differences are small and point both ways; the
> contract matrix is identical q4 against q8 at every size. The 3B is far ahead of
> gemma4:e2b and e4b on grounding and extraction; the 32B trails the newer qwen3.x and large
> gemma4 models on scene grounding and misses one invoice item. The 7B places all five
> invoice name boxes in band at an IoU of 0.33–0.36, the case ADR 0012 rule 11's caveat
> describes. 6 of 6 cells, 12 rung-suites, 0 OOMs, 0 errors, 0 unconverged, 41 minutes,
> beside production.

## Provenance

- **Date / host:** 2026-09-13 15:54:53–16:35:43 local, CUDA box (RTX PRO 6000 Blackwell,
  97.9 GB), shared throughout with the production endpoint on `:11497`
  (`0.33.2-dynres-5-g2b95b4a`, the teacher-v3 rescue legs, 23.7–26.5 GB) and about 12 GB of
  standing services (two Triton servers, three SAM backends and smaller tenants). GPU 0 read
  38.1 GB used at launch and 39.0 GB before the 32B cells.
- **Build under test**, one container on `:11517`:

  | campaign prefix | image | version stamp | ran | wall |
  |---|---|---|---|---|
  | `q25vl_1_` | `maxusai/ollama:sync-0.34.0-main` | `0.34.0-dynres-0-gcf2ad41` | 15:54–16:35 | 41 min |

  This is the `v0.34.0-dynres` tag's build: the deploy candidate `sync-0.34.0` with a Go-only
  binary built from the tag, its native payload byte-identical to the candidate's (5,392
  files). llama.cpp is `b10760`. The driver's `TAG_PREFIX=q25vl_` writes tags
  `q25vl_1_<model>_thinkfalse`, so the generators take `--prefix q25vl_1_`.
- **Reference:** `ggml034_1_1_` (2026-09-11 23:31 to 2026-09-12 00:41,
  `maxusai/ollama:sync-0.34.0-swap`, `0.33.3-dynres-26-gba2eb4f`, `:11516`), the v0.34.0
  fold's own eight-model GGUF campaign
  ([task doc](tasks/upstream-sync-0.34.0.md)). Its tables are re-rendered at the end of this
  document with the current generator, so `name_bbox in-band` reads `x/5` (ADR 0012 rule 11);
  no value changed. The two campaigns are separate tables under separate footers, because a
  single table mixing campaigns is not publishable (rule 10). The GGUF path does compare: no
  file under `llama/`, `llm/` or `cmake/`, and none of `LLAMA_CPP_VERSION`, `Dockerfile`,
  `CMakeLists.txt` or `CMakePresets.json`, changed between `ba2eb4f` and `cf2ad41f`.
  llama.cpp is `b10760` in both, and the qwen25vl gate below was already present at `ba2eb4f`.
- **Models:** six GGUF quantisations of `qwen2.5vl` — `3b` (3.8B) at `q4_K_M` and `q8_0`,
  `7b` (8.3B) at both, `32b` (33.5B) at both. `7b-q8_0` and `32b-q8_0` were pulled from the
  Ollama registry for this campaign (11 and 43 minutes); the other four were already in the
  store. qwen2.5vl has no thinking mode, so think off is the only arm.
- **The fp16 mitigation:** every runner carried `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`, which
  `applyArchServerEnvs` in `llm/llama_server.go` injects for `modelArch == "qwen25vl"`. The
  architecture is read from GGUF metadata, so every tag and quant is covered. It closes the
  fp16-accumulate overflow in the vision tower that returns `?`×31 and poisons the runner slot
  (PR #214, upstream ollama#18070;
  [diagnosis](qwen25vl-3b-poison-image-garbage-decode.md), [knob](qwen25vl-cublas-f32-env.md)).
  The preflight `poison_probe` passes on this build's payload, and no cell here shows the
  signature. `visionServerArgs` also passes `--image-min-tokens 1024`, upstream mtmd's floor
  for Qwen-VL grounding.

  > **Correction (2026-09-19).** "every tag and quant is covered" is true of this campaign
  > and false as a general claim, and it is corrected here rather than rewritten: all six
  > cells above ran registry `qwen2.5vl` tags, whose blobs carry
  > `general.architecture = "qwen25vl"`, so every runner in this document did carry the env
  > and no measurement here moves. What the sentence over-reached on is coverage beyond the
  > registry. `general.architecture` records the converter, not the model: llama.cpp's
  > converter maps Qwen2-VL, Qwen2.5-VL and Qwen2.5-Omni's thinker onto one
  > `MODEL_ARCH.QWEN2VL`, so a **self-converted** Qwen2.5-VL GGUF says `"qwen2vl"` and the
  > gate as it stood on 2026-09-13 never fired for it. Verified on a local conversion of
  > `allenai/olmOCR-2-7B-1025` (text GGUF `"qwen2vl"`, mmproj
  > `clip.projector_type = "qwen2.5vl_merger"`) and reported the same on community
  > `richardyoung/olmocr2:7b-q8`. The gate now matches both spellings — see
  > [the knob doc](qwen25vl-cublas-f32-env.md) for the widened scope and what it costs
  > genuine Qwen2-VL.
- **Endpoint / sampling:** `run_engine_compare.sh` defaults — `/api/chat`, think off at
  temperature 0 (ADR 0029), `num_predict` 2200, ladder from `num_ctx` 8192 to 65536, one
  model at a time, cold restart per cell, `OLLAMA_MAX_LOADED_MODELS=1`,
  `OLLAMA_GPU_OVERHEAD` 16 GiB.
- **Scripts:** `claude-scratch/gate-q25vl.sh`, chained behind `claude-scratch/pull-q25vl.sh`
  by `wait-pull-then-gate.sh`; render `claude-scratch/render-q25vl2.sh`. Every table below is
  generator output (ADR 0012 rule 1: T1 `summarize_engine_compare.py`, T2
  `summarize_head_to_head.py`, `summarize_contract_matrix.py`), rendered after the driver's
  DONE marker (rule 8). Driver log `preflight-runs/q25vl_thinkfalse.log`, runner log
  `preflight-runs/q25vl-runner.log`, render `preflight-runs/q25vl-render.md`. The score files
  sit beside the generators in `vision-suite/` and are gitignored.

## Driver summary

From the driver log's `#####` markers:

| cell | rung 8192 | rung 16384 | wall | OOMs | errors |
|---|---|---|---|---|---|
| `3b-q4_K_M` | 5:03 | 0:51 | 5:54 | 0 | 0 |
| `3b-q8_0` | 2:48 | 1:05 | 3:53 | 0 | 0 |
| `7b-q4_K_M` | 2:47 | 0:49 | 3:36 | 0 | 0 |
| `7b-q8_0` | 3:31 | 1:03 | 4:34 | 0 | 0 |
| `32b-q4_K_M` | 7:21 | 1:50 | 9:11 | 0 | 0 |
| `32b-q8_0` | 10:47 | 2:55 | 13:42 | 0 | 0 |

The first cell's 8192 rung includes the container's cold start.

On every cell the same three arms capped at 8192: `multi_3img`, `multi_3img_anchored` and
`bbox_contract_anchored` (the contract matrix's `bcanchored`). Three images at the 1024-token
floor plus a `num_predict` of 2200 do not fit in 8192. The ladder re-ran those three at 16384
(ADR 0012 convention 9), skipped the 24 arms already scored, and every cell converged. That
is why every row reads `8192/16384 ⚠`: the generator's mixed-window flag, not a defect. T2
carries each arm's own window.

**Reading note.** The driver rewrites a cell's score file after every arm, so between rungs
the file is well-formed and wrong: the capped arms read as failed answers. Read mid-ladder,
`3b-q4_K_M` appeared to fail all three multi-image questions; its converged result is q1 ❌,
q2 ✅, q4-bbox ✅. Only the rendered tables, after DONE, are results.

## q4_K_M against q8_0

- **32B: identical except one IoU.** Every quality cell matches, including the four-of-five
  invoice, the fine-text tiers and both multi-image rows, except document `name_bbox` IoU,
  0.780 against 0.770. q8_0 generates at 21 tok/s against 34 and takes 23.3 s per request
  against 14.2.
- **7B: two small moves, opposite directions.** Scene IoU 0.900 → 0.914, `name_bbox` IoU
  0.364 → 0.327; everything else matches. 74 tok/s against 108.
- **3B: a handful of single-cell flips, both ways.** Scene IoU 0.798 → 0.800, `name_bbox` IoU
  0.723 → 0.779 and fine text 7 px 0 → 1 favour q8_0; serial ✅ → ❌ does not; each
  multi-image row trades one question for another. 117 tok/s against 205.
- **The contract matrix is identical q4 against q8 at every size**: 3B 0 of 8, 7B 3 of 8,
  32B 5 of 8.
- **Speed.** q8_0 costs 31–43 % of generation speed at every size. The direction is
  consistent and expected, since q8_0 weights are roughly 1.7× larger and decode is
  memory-bound, but the exact percentages were measured on a shared card.

On llama.cpp, think-off cells are bit-reproducible per payload, backend, budget and image
(ADR 0012 §4, whose 2026-09-12 amendment confines the non-reproducibility finding to MLX). So
these are real differences between the quants on these images, not run-to-run noise. Each
test is one image, and how far the differences generalise is not measured.

## Against the fold's GGUF models

Read across the two tables; each keeps its own footer.

- **The 3B against the small gemma4 models.** Scene IoU 0.798–0.800 against 0.354
  (`gemma4:e4b-it`) and 0.061 (`gemma4:e2b-it`); invoice `5/5 · 5/5 · ✅` against
  `5/5 · 5/5 · ❌` and `1/5 · 1/5 · ✅`; fine text 14–15 of 20 against 11 and 0. Its weak
  spot is the contract matrix, 0 of 8, the same as e2b.
- **The 32B against the large models.** Scene IoU 0.928 trails `qwen3.8:27b` (0.977),
  `qwen3.6:35b-a3b` (0.974), `gemma4:26b-a4b-it` (0.972) and `gemma4:31b-it` (0.965), and beats
  `nemotron3:33b` (0.862, 0.870). It is the only model of 27B or more in either campaign that
  finds four of five invoice items rather than five. Fine text, 16 of 20, ties qwen3.6, behind
  gemma4:31b (19) and 26b (18). Contract, 5 of 8, ties gemma4:26b and qwen3.6, behind qwen3.8
  (7) and nemotron3 (6). It is the only qwen2.5vl size that answers every multi-image question
  on both rows.
- **The 7B's document geometry.** `name_bbox in-band` is `5/5` on both quants, but the IoU
  behind it is 0.364 and 0.327, against 0.723–0.779 for the 3B and 0.770–0.780 for the 32B.
  The band test counts a box that lands in the right region; the IoU shows these boxes are
  poorly placed within it. This is the caveat ADR 0012 rule 11 records.
- **Throughput does not compare across the two tables.** The reference campaign's host ran
  at a load average near 100 beside a teacher leg and an image build; this one ran beside the
  rescue legs and the standing services. Within each table, the rows ran under one set of
  conditions.

## Not measured here

- Think-on: qwen2.5vl has no thinking mode.
- MLX: no MLX qwen2.5vl tags were in the campaign.
- Repeats: one run per cell. On llama.cpp that is a deterministic draw, but each test is still
  a single image.

## Tables — generator output, `render-q25vl2.sh`, rendered 2026-09-13 16:36:00

## T1 — campaign `q25vl_1_`

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| qwen2.5vl:3b-q4_K_M | GGUF | 8192/16384 ⚠ | 0.798 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| qwen2.5vl:3b-q8_0 | GGUF | 8192/16384 ⚠ | 0.800 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 5/5 |
| qwen2.5vl:7b-q4_K_M | GGUF | 8192/16384 ⚠ | 0.900 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| qwen2.5vl:7b-q8_0 | GGUF | 8192/16384 ⚠ | 0.914 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| qwen2.5vl:32b-q4_K_M | GGUF | 8192/16384 ⚠ | 0.928 | 6/6 · 6/6 · 6/6 | ✅ | 4/5 · 4/5 · ✅ | 4/5 |
| qwen2.5vl:32b-q8_0 | GGUF | 8192/16384 ⚠ | 0.928 | 6/6 · 6/6 · 6/6 | ✅ | 4/5 · 4/5 · ✅ | 4/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5vl:3b-q4_K_M | GGUF | 8192/16384 ⚠ | 4 | 4 | 4 | 2 | 0 | ❌ q1_right | ❌ q4_bbox_hit | — | 465 | 205 | 36630 | 2.4 | 1530 |
| qwen2.5vl:3b-q8_0 | GGUF | 8192/16384 ⚠ | 4 | 4 | 4 | 2 | 1 | ✅ q1 + q2 + q4-bbox | ❌ q1_right | — | 456 | 117 | 39253 | 4.0 | 901 |
| qwen2.5vl:7b-q4_K_M | GGUF | 8192/16384 ⚠ | 4 | 4 | 4 | 2 | 1 | ❌ q4_bbox_hit | ❌ q4_bbox_hit | — | 415 | 108 | 33913 | 3.9 | 916 |
| qwen2.5vl:7b-q8_0 | GGUF | 8192/16384 ⚠ | 4 | 4 | 4 | 2 | 1 | ❌ q4_bbox_hit | ❌ q4_bbox_hit | — | 414 | 74 | 42590 | 5.7 | 631 |
| qwen2.5vl:32b-q4_K_M | GGUF | 8192/16384 ⚠ | 4 | 4 | 3 | 3 | 2 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 479 | 34 | 14996 | 14.2 | 254 |
| qwen2.5vl:32b-q8_0 | GGUF | 8192/16384 ⚠ | 4 | 4 | 3 | 3 | 2 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 479 | 21 | 16911 | 23.3 | 155 |

Provenance (from score files): host(s) http://127.0.0.1:11517 · build(s) 0.34.0-dynres-0-gcf2ad41 · think=false

## Contract matrix, think-off, campaign `q25vl_1_`

## Contract matrix (`contract_followed`), think=false

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5vl:3b-q4_K_M | GGUF | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 8192/16384 ⚠ |
| qwen2.5vl:3b-q8_0 | GGUF | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 8192/16384 ⚠ |
| qwen2.5vl:7b-q4_K_M | GGUF | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | 8192/16384 ⚠ |
| qwen2.5vl:7b-q8_0 | GGUF | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | 8192/16384 ⚠ |
| qwen2.5vl:32b-q4_K_M | GGUF | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192/16384 ⚠ |
| qwen2.5vl:32b-q8_0 | GGUF | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192/16384 ⚠ |

`error` = the arm ran and errored (OOM, transport, HTTP 500), so there is no contract to judge. `cap` = generation stopped at the `num_predict` cap rather than finishing, so the cell carries no score (ADR 0012 rule 8). The cap is a separate limit from the `num_ctx` window.

## T2 — qwen2.5vl:3b think=false: q4_K_M against q8_0

| test | metric | q25vl_1_qwen2_5vl_3b-q4_K_M_thinkfalse | q25vl_1_qwen2_5vl_3b-q8_0_thinkfalse |
|---|---|---|---|
| scene | bbox IoU | 0.798 (8192) | 0.800 (8192) |
| scene | labels / serial | 6/6, ✅ | 6/6, ❌ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.723 (8192) | 0.779 (8192) |
| fine text | 22/16/12/9/7 px | 4/4/4/2/0 (8192) | 4/4/4/2/1 (8192) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ❌ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ❌ 5/5 (16384) | ❌ ✅ ✅ 5/5 (16384) |
| throughput | gen tok/s | 205 | 117 |
| throughput | prefill tok/s | 36630 | 39253 |
| latency | s/req (unique image) | 2.4 | 4.0 |
| latency | req/h (serial) | 1530 | 901 |

Provenance (from score files): host(s) http://127.0.0.1:11517 · build(s) 0.34.0-dynres-0-gcf2ad41

## T2 — qwen2.5vl:7b think=false: q4_K_M against q8_0

| test | metric | q25vl_1_qwen2_5vl_7b-q4_K_M_thinkfalse | q25vl_1_qwen2_5vl_7b-q8_0_thinkfalse |
|---|---|---|---|
| scene | bbox IoU | 0.900 (8192) | 0.914 (8192) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.364 (8192) | 0.327 (8192) |
| fine text | 22/16/12/9/7 px | 4/4/4/2/1 (8192) | 4/4/4/2/1 (8192) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ❌ 5/5 (16384) | ✅ ✅ ❌ 5/5 (16384) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ❌ 5/5 (16384) | ✅ ✅ ❌ 5/5 (16384) |
| throughput | gen tok/s | 108 | 74 |
| throughput | prefill tok/s | 33913 | 42590 |
| latency | s/req (unique image) | 3.9 | 5.7 |
| latency | req/h (serial) | 916 | 631 |

Provenance (from score files): host(s) http://127.0.0.1:11517 · build(s) 0.34.0-dynres-0-gcf2ad41

## T2 — qwen2.5vl:32b think=false: q4_K_M against q8_0

| test | metric | q25vl_1_qwen2_5vl_32b-q4_K_M_thinkfalse | q25vl_1_qwen2_5vl_32b-q8_0_thinkfalse |
|---|---|---|---|
| scene | bbox IoU | 0.928 (8192) | 0.928 (8192) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 4/5, 4/5, ✅, ✅ | 4/5, 4/5, ✅, ✅ |
| document | name_bbox IoU | 0.780 (8192) | 0.770 (8192) |
| fine text | 22/16/12/9/7 px | 4/4/3/3/2 (8192) | 4/4/3/3/2 (8192) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) |
| throughput | gen tok/s | 34 | 21 |
| throughput | prefill tok/s | 14996 | 16911 |
| latency | s/req (unique image) | 14.2 | 23.3 |
| latency | req/h (serial) | 254 | 155 |

Provenance (from score files): host(s) http://127.0.0.1:11517 · build(s) 0.34.0-dynres-0-gcf2ad41

## Reference — campaign `ggml034_1_1_`, re-rendered 2026-09-13 with the current generator

The v0.34.0 fold's own GGUF campaign, from its score files. Re-rendering changed only how
`name_bbox in-band` is printed (`4/5` rather than `4`, ADR 0012 rule 11); every value is as
first rendered on 2026-09-12.

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | 8192 | 0.965 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 8192 | 0.972 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:e4b-it-q4_K_M | GGUF | 8192 | 0.354 | 3/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ❌ | 2/5 |
| gemma4:e2b-it-q4_K_M | GGUF | 8192 | 0.061 | 1/7 · 6/6 · 6/6 | ✅ | 1/5 · 1/5 · ✅ | 0/5 |
| qwen3.8:27b-q4_K_M | GGUF | 8192 | 0.977 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 8192 | 0.974 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| nemotron3:33b-q4_K_M | GGUF | 8192 | 0.862 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| nemotron3:33b-q8 | GGUF | 8192 | 0.870 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 4/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 538 | 34 | 383 | 20.3 | 178 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 536 | 69 | 598 | 10.6 | 340 |
| gemma4:e4b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 3 | 0 | 0 | ❌ q2_right | ✅ q1 + q2 + q4-bbox | — | 443 | 63 | 2896 | 7.6 | 475 |
| gemma4:e2b-it-q4_K_M | GGUF | 8192 | 0 | 0 | 0 | 0 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 630 | 74 | 1363 | 9.7 | 372 |
| qwen3.8:27b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 2 | 1 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 544 | 44 | 1392 | 14.4 | 251 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 2 | 2 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 544 | 27 | 1316 | 22.4 | 160 |
| nemotron3:33b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 54 | 2056 | 10.9 | 331 |
| nemotron3:33b-q8 | GGUF | 8192 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 68 | 2492 | 8.6 | 421 |

Provenance (from score files): host(s) http://127.0.0.1:11516 · build(s) 0.33.3-dynres-26-gba2eb4f · think=false

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
