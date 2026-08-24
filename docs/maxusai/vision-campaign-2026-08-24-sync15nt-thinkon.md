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
  siblings). Driver logs: `preflight-runs/vsuite_nt_thinkon.log` and
  `vsuite_nt_qwen36.log` on the 8 TB array; the ENOSPC-killed first attempt is
  preserved as `*.enospc-2026-08-23.log`.
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
run-to-run variance it needs an n ≥ 3 repeat before it means anything.

## Limitations

- n = 1 per cell; ADR 0023's stochasticity finding applies to every think-on
  number here, capped and converged alike.
- Mixed-rung cells (⚠) are not throughput-comparable; read tok/s only within a
  single-rung row.
- `gemma4:12b-nvfp4`'s ceiling is operator-set at 65536; the default-ladder
  131072 rung was attempted and produced only timeouts (logged, no scores).
- Think-off cells for the gemma4/qwen3.8/nemotron family are the `sync15_1_`
  tags from 2026-08-21/22, rendered with the same summarizer against
  `--prefix sync15_1_`; they were not re-run in this campaign.
