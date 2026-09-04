# Vision campaign 2026-08-24: nvfp4 think-on on the v0.32.15 sync, thrashing check off (sync15nt)

The think-on half of the sync-0.32.15 engine-parity campaign, re-run with MLX's
CUDA graph-cache thrashing check disabled after the first attempt lost 17 of 27
arms on `gemma4:12b-nvfp4` to `cudaGraphAddDependencies` 500s — the fuse
diagnosed in `mlx-thrash-check-masks-as-cudagraph.md` (#211) and fixed at the
runner in #212. Also the first valid `qwen3.6:35b-a3b-nvfp4` cells in both think
modes (the original attempt died of host `ENOSPC`, unrelated to any model).
Think-off cells for the rest of the family are the `sync15_1_` tags measured
2026-08-21/22 on the same image and are not re-run here.

> **Think-on at `temperature 0` is off-policy** ([ADR 0023](adr/0023-think-mode-is-per-model-and-measured-on-policy.md),
> [runaway-reasoning-under-think.md](runaway-reasoning-under-think.md)): greedy
> decoding can by itself prevent reasoning from terminating, stochastically —
> the same arm can converge in 1.7 K tokens on one run and cap at 24 K the next.
> The loop/cap rates below are properties of *(model, quant, greedy decoding)*,
> not model ceilings; converged-arm scores remain valid comparisons against the
> other `temperature 0` runs in this series.

## Provenance

- Image `maxusai/ollama:sync-0.32.15` (build `0.32.14-dynres-108-g76918a7`, MLX
  0.32.1-21-g27fec90), container `vsuite` on `127.0.0.1:11502`, RTX PRO 6000
  Blackwell, `OLLAMA_MAX_LOADED_MODELS=1`, and
  `MLX_ENABLE_CACHE_THRASHING_CHECK=0` set on the container — required on images
  built before #212; from that commit on, the runner defaults the check off and
  an operator export is respected.
- Raw results: `vision-suite/scores_sync15nt_1_*.json` (+ `ft_`/`resp_`/`think_`
  siblings) — **on-host and untracked**: six scores files, matched by the
  `scores_*.json` rule in `vision-suite/.gitignore`, so they are not in this
  repo and this document is the durable record. Driver logs:
  `preflight-runs/vsuite_nt_thinkon.log` and `vsuite_nt_qwen36.log` on the 8 TB
  array (`/mnt/8TB_SN850X_RAID1_BTRFS/preflight-runs/`); the ENOSPC-killed
  first attempt is preserved as `*.enospc-2026-08-23.log`, three files. Logs
  and scores files re-checked 2026-09-04, all present.
- Wall clock (2026-08-23/24): 12b 15:22-22:11 (see the 131072 note), 26b
  22:11-23:33, 31b 23:33-00:01, qwen3.8 00:01-00:23, qwen3.6 think-off
  00:23-00:35, qwen3.6 think-on 00:35-05:16. Zero server ERROR lines and zero
  runner panics across the ~14 h.

## Method

- [`vision-suite/run_engine_compare.sh`](vision-suite/run_engine_compare.sh)
  with a cold `docker restart` per cell, card sampling presets pinned to
  `temperature 0` (`sampling_source: card:<model>+temp0`), `ENDPOINT=chat`,
  suite defaults otherwise. The restart hook additionally requires 8 GiB free on
  `/` before each cell (prune dangling images, else pause) — the first attempt
  was killed mid-write by a full root filesystem, which is what a 0-byte
  `scores_` file means.
- **Context ladder**: think-on starts at `num_ctx` 16384 with
  `num_predict = num_ctx − 8192`, and after each rung exactly the still-capped
  arms re-run one rung higher (16384 → 32768 → 65536 → …).
- **Ceiling**: the first cell ran with the default `CTX_MAX=131072`. Its 131072
  rung produced nothing but 1800 s `HTTP_TIMEOUT` 500s — three per arm, ~90 min
  per arm, no data — exactly as the suite README warns ("do not escalate
  `num_ctx` to chase this"). The campaign was stopped at that rung boundary
  (22:11, marker `##### STOPPED BY OPERATOR` in the log) and every remaining
  cell ran with `CTX_MAX=65536`. `gemma4:12b-nvfp4`'s capped arms therefore
  stand at the 65536 rung.

  **That stop is now permanent, not a pause.** As of 2026-08-30
  `(gemma4:12b-nvfp4, "on")` is declared in `DESCOPED_CELLS`
  (`summarize_engine_compare.py`): the cell is closed by operator decision
  rather than left open, `run_engine_compare.sh` skips it before any rung, and
  the 131072 ceiling will not be attempted. What the measurement would have
  bought is available cheaper — 31b converges at rung 1 here, qwen3.8 likewise.
  The evidence below is the argument that declaration points back to, not a
  backlog item. **think-off for this tag is unaffected** and still renders in
  the cross-engine view.
- Reproduce:

  ```sh
  cd docs/maxusai/vision-suite
  RESTART_CMD=<cold-restart hook> TAG_PREFIX="sync15nt_" CTX_START=8192 CTX_MAX=65536 \
    THINK_MODES="on" MODELS="gemma4:12b-nvfp4 gemma4:26b-nvfp4 gemma4:31b-nvfp4 qwen3.8:27b-nvfp4" \
    ./run_engine_compare.sh http://127.0.0.1:11502
  RESTART_CMD=<hook> TAG_PREFIX="sync15nt_" CTX_START=8192 CTX_MAX=65536 \
    MODELS="qwen3.6:35b-a3b-nvfp4" ./run_engine_compare.sh http://127.0.0.1:11502
  python3 summarize_engine_compare.py --think on --prefix sync15nt_1_ \
    gemma4:12b-nvfp4 gemma4:26b-nvfp4 gemma4:31b-nvfp4 qwen3.8:27b-nvfp4 qwen3.6:35b-a3b-nvfp4
  python3 summarize_engine_compare.py --think false --prefix sync15nt_1_ qwen3.6:35b-a3b-nvfp4
  ```

  Re-run today those commands do not reproduce this document exactly: the
  driver checks `DESCOPED_CELLS` before every rung, so the first one measures
  three of its four models, and the summarizer prints `⚠ DESCOPED` in place of
  the `gemma4:12b-nvfp4` think-on rows instead of rendering them. The rows
  below were rendered 2026-08-24, before that declaration existed.

## Reading the ladder: what "converged 25/27" means

Each cell runs the full 27-arm battery (the `bboxm_*` declaration grid, the
`bbox_contract_*` variants, `scene_single*`, `document_single`, `multi_3img*`,
`finetext*`). Per arm:

- **converged** — the model finished its answer within the generation budget at
  some rung: `done_reason: "stop"`, `eval_count < num_predict`. The row is a
  real measurement, and `num_ctx` in the scores records the rung it was achieved
  at. *Converged does not mean correct* — a converged arm can still score 1/6
  (see Findings §6).
- **capped** — `eval_count == num_predict` with `done_reason: "length"`: every
  token went into an unclosed thinking block and the answer is empty. This is a
  budget artifact, not a quality result; the ladder re-runs exactly these arms
  one rung higher while already-scored arms are skipped.
- **NOT CONVERGED** — still capped at `CTX_MAX`. Recorded as a finding about
  *(model, quant, greedy decoding)* at that budget and left at that rung.

So "converged 25/27" reads: 25 arms produced complete, scored answers; 2 never
terminated within 57,344 generated tokens and have no answer to score. In the
tables below, "capped" cells and the ⚠ on mixed `num_ctx` mean exactly this —
and cells measured at different rungs are not comparable on throughput, because
KV size moves decode speed.

Per-cell ladder outcome:

| Cell (think=on unless noted) | converged | NOT CONVERGED at ceiling | arms per rung |
|---|---|---|---|
| gemma4:12b-nvfp4 | 14/27 | 13 | 16384: 10 · 32768: 2 · 65536: 15 |
| gemma4:26b-nvfp4 | 25/27 | 2 | 16384: 19 · 32768: 5 · 65536: 3 |
| gemma4:31b-nvfp4 | 27/27 | 0 | 16384: 27 |
| qwen3.8:27b-nvfp4 | 27/27 | 0 | 16384: 27 |
| qwen3.6:35b-a3b-nvfp4 think=false | 27/27 | 0 | 8192: 27 |
| qwen3.6:35b-a3b-nvfp4 | 24/27 | 3 | 16384: 10 · 32768: 8 · 65536: 9 |

**That table is derived, not generator output.** All three data columns were
read out of the scores files by the orchestrator: `converged` /
`NOT CONVERGED` from the harness's own `was_capped` (done_reason-first, SPEC
H5 — which is why an arm that overshoots its budget and still reports
`done_reason: "stop"` counts as converged), and `arms per rung` from each
arm's final `req_num_ctx`, counting all 27 arms whether they converged at that
rung or capped there. **The veto is the files themselves** —
`scores_sync15nt_1_<cell>.json`, fields `req_num_ctx` and `done_reason`,
re-checked against them 2026-09-04. Where the prose below disagrees with those
fields, the fields win.

`gemma4:12b-nvfp4` think=on is a **descoped** cell as of 2026-08-30, not an
open one (see the Ceiling note above). Its 14/27, and the 13 arms standing at
65536, are the CUDA-host counterpart of the evidence the declaration cites —
that evidence is the Apple Silicon cell measured in
[the 2026-08-28 campaign](vision-campaign-2026-08-28-mlx0330-nvfp4.md), which
reached 7/27 at the same rung and was priced at ~9.5 h. Both point the same
way. Neither is a measurement to be continued.

## Throughput columns are single-run figures (caveat added 2026-09-04)

**Read this before quoting any `Gen tok/s`, `s/req` or `req/h` number in this
document, the cross-engine tables included.** The score columns are unaffected.

Every timing cell below is n = 1. On 2026-08-31 a 38% decode regression
reported from cells exactly like these was **retracted**: three repeats of the
same arms on the same binary in the same session put the campaign's figure
1.7× below the slowest repeat — §3 of
[the 0.33.2 campaign](vision-campaign-2026-08-31-mlx0332-nvfp4.md)
([#257](https://github.com/MaxusAI/ollama/pull/257)). The lesson is not about
that build. It is that every argument available *within* one cell — 27 arms
sharing one server process, one machine state, one thermal condition — can
establish that a cell ran slow and cannot separate "this cell ran slow" from
"this stack is slow", because the confound is constant across every arm in it.
Each throughput number here was produced that way.

The spread measured behind that retraction (~2×, state-dependent) is from the
Apple Silicon host. **No equivalent repeat has been run on this CUDA host**, so
the size of the spread *here* is unmeasured — a reason to hold these columns
more loosely, not less. Read them as an order-of-magnitude characterisation of
a served stack; never as a property of a build, an engine or a kernel.

When a throughput figure matters, the mechanism is
[#258](https://github.com/MaxusAI/ollama/pull/258)'s: the figure is a
**trigger for a targeted post-campaign re-run of that one cell** — serve the
build standalone, run the same arms 3× back to back, compare against the same
arms in the reference cell. It is not `REPEATS=n`. `rep` is the innermost loop
in the driver, so reps nested inside a cell share its machine state and buy a
tight interval around whatever that cell did that night, at campaign-wide cost.

## Results — think=on

### Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 16384/65536 ⚠ | capped | capped | capped | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-nvfp4 | **MLX** | 16384/32768/65536 ⚠ | **0.970** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.964** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | **0.999** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384/65536 ⚠ | capped | capped | capped | 5/5 · 5/5 · ✅ | 4 |

### Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Gen tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 16384/65536 ⚠ | capped | capped | capped | capped | capped | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | capped | ≥57344 ⚠ | 72 | 2604 | capped | capped |
| gemma4:26b-nvfp4 | **MLX** | 16384/32768/65536 ⚠ | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | capped | — | 6008 | 77 | 88 | 97.6 | 37 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1977 | 44 | 184 | 54.3 | 66 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 1 | 1 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1030 | 29 | 4381 | 35.7 | 101 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384/65536 ⚠ | 4 | 4 | 4 | 2 | 2 | ✅ q1 + q2 + q4-bbox | ❌ q4_bbox_hit | capped | ≥57344 ⚠ | 76 | 4052 | capped | capped |

Provenance (from score files): host(s) http://127.0.0.1:11502 · build(s) 0.32.14-dynres-108-g76918a7 · think=on

## Results — qwen3.6:35b-a3b-nvfp4 think=false

### Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 8192 | **0.963** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |

### Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 8192 | 4 | 4 | 4 | 2 | 1 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 538 | 26 | 8854 | 21.0 | 171 |

Provenance (from score files): host(s) http://127.0.0.1:11502 · build(s) 0.32.14-dynres-108-g76918a7 · think=false

## Cross-engine view — GGUF vs MLX on the CUDA host

GGUF rows below are the `sync15_1_` cells (2026-08-21/22, same image, host and
fixtures; llama-server engine). GGUF requests never touch the MLX runner, so the
thrashing fuse that invalidated the first MLX think-on batch does not affect
them. MLX think-on rows are this campaign; MLX think-off rows are `sync15_1_`
except `qwen3.6:35b-a3b-nvfp4` (this campaign). Same-family pairs differ in
quantization (nvfp4 vs q4_K_M) — the quant is part of each engine's serving
format, so the pairs compare *served stacks*, not bare kernels. Note
`gemma4:26b-nvfp4` (dense) vs `gemma4:26b-a4b` (MoE) is **not** a same-model
pair.

### think=false — scene grounding + document

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 8192 | **0.956** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-nvfp4 | **MLX** | 8192 | **0.973** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 8192 | 0.973 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-nvfp4 | **MLX** | 8192 | **0.962** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 0.962 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:e4b-it-q4_K_M | GGUF | 8192 | 0.341 | 3/7 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ❌ | 3 |
| gemma4:e2b-it-q4_K_M | GGUF | 8192/131072 ⚠ | 0.087 | 1/8 · 6/6 · 6/6 | ✅ | 1/5 · 1/5 · ✅ | 0 |
| qwen3.8:27b-nvfp4 | **MLX** | 8192 | **0.999** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.8:27b-q4_K_M | GGUF | 8192 | 0.977 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 8192 | **0.963** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 16384 | 0.975 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 0.870 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q8 | GGUF | 8192 | 0.870 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 4 |

### think=false — fine-text OCR + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 8192 | 4 | 4 | 3 | 0 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 535 | 22 | 1001 | 26.1 | 138 |
| gemma4:26b-nvfp4 | **MLX** | 8192 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 547 | 30 | 589 | 21.3 | 169 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 536 | 181 | 1327 | 4.2 | 852 |
| gemma4:31b-nvfp4 | **MLX** | 8192 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 537 | 28 | 880 | 21.1 | 171 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 538 | 52 | 436 | 14.2 | 253 |
| gemma4:e4b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 0 | 0 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 501 | 180 | 3054 | 3.3 | 1080 |
| gemma4:e2b-it-q4_K_M | GGUF | 8192/131072 ⚠ | capped | capped | capped | capped | capped | ❌ q4_bbox_hit | ❌ q4_bbox_hit | — | 697 | 235 | 3446 | 3.5 | 1042 |
| qwen3.8:27b-nvfp4 | **MLX** | 8192 | 4 | 4 | 4 | 2 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 549 | 21 | 8404 | 26.5 | 136 |
| qwen3.8:27b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 2 | 1 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 544 | 70 | 1815 | 9.2 | 391 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 8192 | 4 | 4 | 4 | 2 | 1 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 538 | 26 | 8854 | 21.0 | 171 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 2 | 2 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 548 | 108 | 3323 | 5.8 | 616 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 270 | 5038 | 2.4 | 1485 |
| nemotron3:33b-q8 | GGUF | 8192 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 227 | 4956 | 2.8 | 1288 |

### think=on — scene grounding + document

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 16384/65536 ⚠ | capped | capped | capped | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-nvfp4 | **MLX** | 16384/32768/65536 ⚠ | **0.970** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384/65536 ⚠ | 0.334 | 0/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.964** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 0.962 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:e4b-it-q4_K_M | GGUF | 16384 | 0.292 | 3/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 2 |
| gemma4:e2b-it-q4_K_M | GGUF | 16384 | 0.238 | 4/6 · 6/6 · 5/6 | ✅ | 0/5 · 0/5 · ✅ | 0 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | **0.999** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 0.990 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384/65536 ⚠ | capped | capped | capped | 5/5 · 5/5 · ✅ | 4 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 16384/32768/131072 ⚠ | 0.717 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q4_K_M | GGUF | 16384/32768 ⚠ | 0.862 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q8 | GGUF | 16384/32768 ⚠ | 0.265 | 0/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 5 |

### think=on — fine-text OCR + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 16384/65536 ⚠ | capped | capped | capped | capped | capped | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | capped | ≥57344 ⚠ | 72 | 2604 | capped | capped |
| gemma4:26b-nvfp4 | **MLX** | 16384/32768/65536 ⚠ | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | capped | — | 6008 | 77 | 88 | 97.6 | 37 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384/65536 ⚠ | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 7576 | 173 | 478 | 47.4 | 76 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1977 | 44 | 184 | 54.3 | 66 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1372 | 57 | 165 | 34.3 | 105 |
| gemma4:e4b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 1 | 1 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 2086 | 179 | 1539 | 12.7 | 283 |
| gemma4:e2b-it-q4_K_M | GGUF | 16384 | 0 | 0 | 0 | 0 | 0 | ❌ q4_bbox_hit | ❌ q1_right | — | 1756 | 243 | 1779 | 8.2 | 440 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 1 | 1 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1030 | 29 | 4381 | 35.7 | 101 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 1 | 0 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 1084 | 68 | 1572 | 17.5 | 205 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384/65536 ⚠ | 4 | 4 | 4 | 2 | 2 | ✅ q1 + q2 + q4-bbox | ❌ q4_bbox_hit | capped | ≥57344 ⚠ | 76 | 4052 | capped | capped |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 16384/32768/131072 ⚠ | 4 | 4 | 4 | 2 | 2 | capped | ✅ q1 + q2 + q4-bbox | — | 21058 | 104 | 2407 | 203.6 | 18 |
| nemotron3:33b-q4_K_M | GGUF | 16384/32768 ⚠ | 4 | 4 | 4 | 4 | 0 | ✅ q1 + q2 + q4-bbox | ❌ q1_right, q2_right, q4_bbox_hit | — | 3583 | 271 | 2760 | 14.2 | 253 |
| nemotron3:33b-q8 | GGUF | 16384/32768 ⚠ | 4 | 4 | 4 | 4 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 6223 | 232 | 2354 | 28.0 | 129 |

### Same-weights pairs, at a glance

| pair | mode | scene IoU GGUF / MLX | req/h GGUF / MLX | GGUF req/h advantage |
|---|---|---|---|---|
| gemma4:31b | off | 0.962 / 0.962 | 253 / 171 | 1.5× |
| qwen3.8:27b | off | 0.977 / **0.999** | 391 / 136 | 2.9× |
| qwen3.6:35b-a3b | off | **0.975** / 0.963 | 616 / 171 | 3.6× |
| gemma4:31b | on | 0.962 / **0.964** | 105 / 66 | 1.6× |
| qwen3.8:27b | on | 0.990 / **0.999** | 205 / 101 | 2.0× |

### Reading the cross-engine view

- **Scores: engine parity.** Same-weights pairs agree to within quant-level
  differences, and those cut both ways (nvfp4 ahead on qwen3.8's scene IoU,
  q4_K_M ahead on qwen3.6's). No column shows an engine-level quality gap.
- **Throughput: GGUF wins 1.5–3.6× req/h on this host.** MLX-CUDA's prefill is
  the fastest in the fleet (qwen: 8,400–8,900 tok/s vs GGUF's 1,800–3,300), but
  its short-answer decode runs 21–30 tok/s against GGUF's 52–270, and a
  ~540-token think-off answer is almost all decode. Long think-on generations
  amortize MLX decode to 72–77 tok/s, which narrows the gap (1.6–2.0×) without
  closing it. **Every ratio in this bullet and in the pairs table above comes
  from single runs** — see the 2026-09-04 caveat. The direction is consistent
  across five pairs and both think modes, which is what the bullet rests on;
  no individual ratio is a settled number, and the smallest ones are the ones a
  repeat could most plausibly move. Settling one means a targeted 3× re-run of
  that pair, not a closer reading of this table.
- **Think-on hazards are cross-engine, not an MLX property.** MLX loops (12b,
  qwen3.6-nvfp4); GGUF has its own — `gemma4:26b-a4b` scene falls to 0.334,
  `qwen3.6-q4_K_M` spends 21 K think tokens for 18 req/h at the 131072 rung,
  `nemotron3-q8` scene 0.265. Only 31b and qwen3.8 are think-on-safe on either
  engine, and neither gains a metric from thinking (ADR 0022/0023 hold).
- Mixed-rung rows (⚠) are not throughput-comparable; read tok/s and req/h only
  within single-rung rows.

No decision is recorded here. If the serving call "GGUF serves short-answer
vision extraction on the CUDA host until MLX-CUDA's per-request decode overhead
is addressed" is adopted, capture it as an ADR citing this section — and, since
that call turns on throughput, on a targeted 3× re-run of the pair it rests on
rather than on these single-run cells alone (2026-09-04 caveat, #258).

## Findings

**1. No regression from the v0.32.15 sync.** `gemma4:31b-nvfp4` is the one cell
with a pre-sync baseline (the three 2026-08-17 `mlxcuda1/2/3` repeats on the
pre-sync MLX image): all 12 comparable arms converged in both, and every metric
is within ±0.005 IoU — the same spread the three pre-sync repeats show among
themselves. The only material mover is `bbox_contract_adv_real`, which
*improved* 1 → 3 hits (IoU 0.026 → 0.401) and stays a hard arm. Against the
same-image `sync15_` run: 18 of 26 comparable arms identical, the rest within
±0.003, and `multi_3img` now converges where it errored.

**2. 31b and qwen3.8 are the disciplined reasoners.** Both converge all 27 arms
at the first rung. `qwen3.8:27b-nvfp4` posts the best scene IoU of the series
(0.999) but the weakest small-text OCR (9 px/7 px: 1/1 vs 31b's 3/3, 26b's
4/3) — a capability difference measured entirely on converged arms.

**3. The loop ranking is 12b ≫ qwen3.6 > 26b, and context does not fix it.**
13 / 3 / 2 arms NOT CONVERGED respectively; every capped arm hit its budget
exactly at every rung offered (8192 → 24576 → 57344). The 131072 experiment on
12b converts caps into timeouts, not answers. Per ADR 0023 the lever is
sampling (`presence_penalty` per the model cards), not context — these cells
quantify the greedy-decoding failure mode, deliberately left on-policy-off.

For `gemma4:12b-nvfp4` think-on the ladder stops here for good: that cell is
descoped as of 2026-08-30, so its 13 standing arms are a **closed** finding
about *(model, quant, greedy decoding)* and not work queued up. 26b's 2 and
qwen3.6's 3 are not descoped and stand as ordinary NOT CONVERGED verdicts,
reopenable by raising `CTX_MAX` on those cells.

**4. qwen3.6:35b-a3b-nvfp4 wants think=off.** Think-off: 27/27 at `num_ctx`
8192, scene IoU 0.963, full boxes/colors, invoice clean, 171 req/h — the
fastest cell in the series. Think-on adds three NOT CONVERGED arms, one
multi-image regression (`q4_bbox_hit` ❌), and costs ~8× the wall clock for no
metric gain anywhere.

**5. The thrashing check stays off.** 32 h earlier the identical 12b cell lost
17/27 arms to `cudaGraphAddDependencies` 500s with per-request runner restarts.
This run: zero server errors, zero panics, one runner start per rung, across
~14 h and every long-decode cell — with the check disabled by container env
(and by runner default from #212 onward). Mechanism and A/B measurements:
`mlx-thrash-check-masks-as-cudagraph.md` (#211).

**6. Open item — clean-but-wrong on 26b's free-form grid.** Two converged arms
(`bboxm_free_anc_named`, `bboxm_free_noanc_pos`) returned well-formed answers
with 1/6 boxes (IoU 0.13/0.12) where the previous same-image run had 6/6
(0.70/0.94), while every `bboxm_pin_*` arm and 12b/31b's free arms are 6/6.
Two of four in one family is a signal, but at `temperature 0` with this much
run-to-run variance it means nothing until it is repeated.

The instrument is the one
[#258](https://github.com/MaxusAI/ollama/pull/258) settled on: a **targeted
post-campaign re-run of this one cell** — serve the build standalone, run
`bboxm_free_anc_named` and `bboxm_free_noanc_pos` 3× back to back, and compare
against the same two arms in the reference cell, which is the control. **Not**
`REPEATS=n`: `rep` is the innermost loop in the driver, so it re-prices every
cell in the campaign to buy repeats of one.

## Limitations

- **n = 1 per cell, deliberately.** The campaign is a quality gate and n = 1 is
  the right setting for it; ADR 0023's stochasticity finding applies to every
  think-on number here, capped and converged alike. Throughput is the column
  n = 1 does not carry — see the 2026-09-04 caveat. The policy that follows is
  not "run bigger campaigns": an anomaly gets a targeted re-run of its own cell
  afterwards.
- Mixed-rung cells (⚠) are not throughput-comparable; read tok/s only within a
  single-rung row — and, per that caveat, not as a build or engine property
  even then.
- `gemma4:12b-nvfp4`'s ceiling is operator-set at 65536; the default-ladder
  131072 rung was attempted and produced only timeouts (logged, no scores).
  Since 2026-08-30 that is policy rather than a per-campaign setting: the
  think-on cell is declared in `DESCOPED_CELLS` and is not re-run at any rung.
  think-off for the same tag is unaffected.
- Think-off cells for the gemma4/qwen3.8/nemotron family are the `sync15_1_`
  tags from 2026-08-21/22, rendered with the same summarizer against
  `--prefix sync15_1_`; they were not re-run in this campaign.
