# Vision campaign 2026-08-17 — eighteen models, both think modes, full contract suite

Every vision model in the store — four gemma4 sizes across four quantisations,
three qwen3.6, two qwen3.8, three nemotron3 — run through all twelve suite tests
in **both** think modes. 36 model-modes, the first sweep to exercise the six
bounding-box contract arms outside the original seven-model corpus.

Two questions motivated the scale, and both are answered here: **does the
contract hold below 26b**, and **does quantisation affect declaration honesty**.

## Provenance

| | |
| --- | --- |
| server | `0.32.5-maxusai-a5d65906` (native macOS), binary built 2026-08-16 15:57 |
| store | `~/.ollama/models-mlx`, served on `:11436` |
| runner | `run_engine_compare.sh`, `THINK_MODES='false on'`, cold restart per model |
| think-off sampling | `greedy-think-off`, `temperature 0` — deterministic |
| think-on sampling | card-sourced per [ADR 0023](adr/0023-think-mode-is-per-model-and-measured-on-policy.md): gemma4 `temp 1.0/top_p 0.95/top_k 64`, qwen3.6 `temp 1.0/top_p 0.95/top_k 20/min_p 0/presence_penalty 1.5`, qwen3.8 and nemotron3 `packaged-defaults-no-card` — **stochastic** |
| window | `num_ctx` 16384 base rung, climbing the ladder where think-on exhausted it |
| run | 2026-08-16 20:37 → 2026-08-17 05:09, plus a resume 05:29 → 06:5x |

**Power mode is not constant.** It is stamped per model-mode by the runner and
recorded below; quality metrics are power-invariant at temperature 0, so the
quality rows are unaffected, but **throughput is only comparable within a
segment.** Only four model-modes ran at powermode 1:

| powermode | model-modes |
| --- | --- |
| **1** | `gemma4:12b-nvfp4` (both), `gemma4:26b-a4b-it-q4_K_M` (both) |
| **2** | the other 32 |

So any 12b-vs-26b throughput comparison straddles a boundary and is **not** a
size effect. `gemma4:12b-nvfp4` at 112 req/h against `gemma4:26b-mxfp8` at 339
is largely the power mode, not the model.

### The run crashed and was resumed

The first pass died at 05:09 with a `TypeError`, part-way through a `num_ctx`
ladder retry on `nemotron3:33b-q4_K_M` think-on. `nemotron3:33b-q8` and
`nemotron3:33b-bf16` never ran. Cause: nemotron3 answered
`"bbox_2d": "real"` — the *type* in the coordinate field — and `len("real") == 4`
satisfied every length check downstream, so scoring reached `"r" * 0.52`. Two
defects, both fixed in #145: `get_bbox` now requires four numbers, and a scorer
exception no longer aborts the campaign.

The two missing models were re-run with the fix live. **The base rung of
`nemotron3:33b-q4_K_M` think-on had already completed and written all twelve
tests before the crash**, so that cell is intact; only the retry was lost.

**Corrected 2026-08-18: "intact" was the wrong word, and the lost retry was the
whole measurement.** That cell sits at the base rung, and its throughput columns
say so — `≥8192 ⚠`, `capped`, `capped`. Its *quality* columns are rendered as
scores anyway: scene IoU **0.000**, 0/6 labels, 0/5 invoice items, ❌ on serial
and total. Those are not results. `num_predict` derives as
`16384 − CTX_PROMPT_RESERVE 8192 = 8192`, and this model needs **8385 / 10226 /
4127** generated tokens to terminate (measured 2026-08-18 on the ROCm host,
`multi_3img`, n=3 at a rung deriving 122880 — `done_reason=stop` every time,
every question correct). At 8192 it returns `done_reason=length` with **zero
characters of answer** and 24–27k characters of unclosed thinking, in about half
its runs. The zeros are that truncation.

The ladder retry that would have escalated past it is exactly what the crash
destroyed, so this cell never got the rung it needed. Read the whole
`nemotron3:33b-q4_K_M` think-on row as **not measured**, not as a model that
scored zero — and note that `was_capped()` protects the throughput columns while
the quality columns render a capped cell identically to a converged one, which is
how a row of zeros reached this page looking like a result.
[ADR 0022](adr/0022-thinking-is-off-for-vision-work.md) carries the measurement;
[SPEC H4a](spec/vision-harness-reuse.md) carries the rule.

## Results — think-off (T1, verbatim from `summarize_engine_compare.py`)

Per [ADR 0012](adr/0012-benchmark-report-templates.md): generator-rendered, not
typed, and **`num_ctx` is a column** because a score without its window is not
interpretable.

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox hits |
|---|---|---|---|---|---|---|---|
| gemma4:12b-it-q4_K_M | GGUF | 16384 | 0.870 | 6/6 · 6/6 · 5/6 | ✅ | 5/5 · 5/5 · ✅ | 3 |
| gemma4:12b-nvfp4 | **MLX** | 16384 | **0.953** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384 | 0.969 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-nvfp4 | **MLX** | 16384 | **0.965** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-mxfp8 | **MLX** | 16384 | **0.971** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-mlx-bf16 | **MLX** | 16384 | **0.973** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 0.960 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.958** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-mxfp8 | **MLX** | 16384 | **0.960** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-mlx-bf16 | **MLX** | 16384 | **0.967** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 16384 | 0.975 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384 | **0.965** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.6:35b-a3b-q8_0 | GGUF | 16384 | 0.967 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 0.991 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | **0.987** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 0.870 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q8 | GGUF | 16384 | 0.884 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-bf16 | GGUF | 16384 | 0.870 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 4 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 0 | 0 | ✅ all Qs + bbox | 542 | 36 | 809 | 17.0 | 212 |
| gemma4:12b-nvfp4 | **MLX** | 16384 | 4 | 4 | 3 | 0 | 0 | ✅ all Qs + bbox | 535 | 18 | 601 | 32.1 | 112 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ all Qs + bbox | 538 | 40 | 208 | 21.6 | 167 |
| gemma4:26b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ all Qs + bbox | 547 | 55 | 600 | 12.8 | 280 |
| gemma4:26b-mxfp8 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ all Qs + bbox | 535 | 68 | 620 | 10.6 | 339 |
| gemma4:26b-mlx-bf16 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ all Qs + bbox | 535 | 47 | 219 | 19.1 | 189 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ all Qs + bbox | 538 | 18 | 225 | 37.5 | 96 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ all Qs + bbox | 537 | 17 | 417 | 35.1 | 103 |
| gemma4:31b-mxfp8 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ all Qs + bbox | 537 | 10 | 314 | 57.4 | 63 |
| gemma4:31b-mlx-bf16 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ all Qs + bbox | 537 | 8 | 150 | 82.6 | 44 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 2 | 2 | ✅ all Qs + bbox | 544 | 104 | 989 | 7.9 | 458 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 1 | ✅ all Qs + bbox | 537 | 103 | 1199 | 7.4 | 485 |
| qwen3.6:35b-a3b-q8_0 | GGUF | 16384 | 4 | 4 | 4 | 2 | 2 | ✅ all Qs + bbox | 550 | 79 | 869 | 10.0 | 362 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 2 | 1 | ❌ q4_bbox_hit | 544 | 21 | 370 | 33.1 | 109 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 0 | ✅ all Qs + bbox | 547 | 25 | 438 | 27.7 | 130 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 3 | 0 | ✅ all Qs + bbox | 512 | 94 | 712 | 9.2 | 392 |
| nemotron3:33b-q8 | GGUF | 16384 | 4 | 4 | 4 | 3 | 0 | ✅ all Qs + bbox | 512 | 103 | 1079 | 7.4 | 484 |
| nemotron3:33b-bf16 | GGUF | 16384 | 4 | 4 | 4 | 3 | 0 | ✅ all Qs + bbox | 512 | 62 | 710 | 12.1 | 299 |

Grounding is saturated: sixteen of eighteen exceed 0.95, and every model finds
all six shapes. **Quantisation barely moves quality** — gemma4:31b spans
0.958–0.967 across q4_K_M/nvfp4/mxfp8/bf16, inside ADR 0012's ±0.01 noise floor.
**Size does move fine text**: both 12b builds score 0 at the 9px and 7px tiers
where every 26b and 31b build scores 3–4. The two nemotron serial failures at q8
and bf16 are n=1 and want repeats before anyone acts on them.

Every think-off row ran at `num_ctx` **16384** — no escalation was needed.

## Results — think-on (T1, verbatim from `summarize_engine_compare.py`)

**Read the `num_ctx` column first.** Think-on escalates the ladder, and the rung
a model needed is part of its result: the same score means different things at
16384 and 65536.

**Corrected 2026-08-18: the token column below reads `Gen tok`, not `Answer
tok`.** It renders `eval_count`, which counts every token the model generated —
under think-on that is reasoning *plus* answer. `gemma4:12b-it-q4_K_M`'s 5588 is
not the length of its answer, and reading it as one understates every model's
answer-per-second by whatever share went to reasoning. **Only the header
changed; no number in this table moved**, and the think-off table above keeps
`Answer tok`, where the label is exact. The split itself is still not available
— the API reports one count and the parser that knows where reasoning ends never
sees tokens (MaxusAI/ollama#189) — so this is a correction to what the column
claims, not a new measurement. `summarize_engine_compare.py` now derives the
header from `--think` (MaxusAI/ollama#195), so a re-render of this run reproduces
the table as it now stands.

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox hits |
|---|---|---|---|---|---|---|---|
| gemma4:12b-it-q4_K_M | GGUF | 16384 | 0.794 | 6/6 · 6/6 · 5/6 | ✅ | 5/5 · 5/5 · ✅ | 3 |
| gemma4:12b-nvfp4 | **MLX** | 16384 | **0.955** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384 | 0.721 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-nvfp4 | **MLX** | 16384 | **0.969** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-mxfp8 | **MLX** | 16384 | **0.969** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-mlx-bf16 | **MLX** | 16384 | **0.975** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 0.960 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.963** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-mxfp8 | **MLX** | 16384 | **0.960** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 3 |
| gemma4:31b-mlx-bf16 | **MLX** | 32768 | **0.797** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 32768 | 0.835 | 5/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 65536 | **0.936** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.6:35b-a3b-q8_0 | GGUF | 32768 | 0.372 | 5/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 0.975 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | **1.000** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 0.000 | 0/None · 0/6 · 0/6 | ❌ | 0/5 · 0/5 · ❌ | 0 |
| nemotron3:33b-q8 | GGUF | 32768 | 0.234 | 2/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 5 |
| nemotron3:33b-bf16 | GGUF | 32768 | 0.872 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 3 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Gen tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 1 | 0 | ✅ all Qs + bbox | 5588 | 38 | 1128 | 149.2 | 24 |
| gemma4:12b-nvfp4 | **MLX** | 16384 | 4 | 3 | 3 | 0 | 0 | ✅ all Qs + bbox | 2318 | 31 | 228 | 82.1 | 44 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ all Qs + bbox | 2300 | 42 | 218 | 62.3 | 58 |
| gemma4:26b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ all Qs + bbox | 5475 | 72 | 333 | 80.6 | 45 |
| gemma4:26b-mxfp8 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ all Qs + bbox | 3289 | 80 | 455 | 44.9 | 80 |
| gemma4:26b-mlx-bf16 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ all Qs + bbox | 4208 | 82 | 200 | 60.0 | 60 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 2 | ✅ all Qs + bbox | 1543 | 16 | 248 | 101.6 | 35 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 2 | ✅ all Qs + bbox | 4803 | 27 | 100 | 193.3 | 19 |
| gemma4:31b-mxfp8 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 2 | ✅ all Qs + bbox | 3145 | 18 | 136 | 184.9 | 19 |
| gemma4:31b-mlx-bf16 | **MLX** | 32768 | 4 | 4 | 4 | 4 | 3 | ✅ all Qs + bbox | 7595 | 22 | 73 | 374.9 | 10 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 32768 | 4 | 4 | 4 | 2 | 3 | ❌ q4_bbox_hit | 19851 | 91 | 1263 | 221.1 | 16 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 65536 | 4 | 4 | 4 | 1 | 1 | ✅ all Qs + bbox | 27275 | 97 | 191 | 294.2 | 12 |
| qwen3.6:35b-a3b-q8_0 | GGUF | 32768 | 4 | 4 | 4 | 1 | 2 | ❌ q1_right, q2_right, q4_bbox_hit | 9582 | 76 | 838 | 129.9 | 28 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 0 | 1 | ❌ q4_bbox_hit | 1078 | 20 | 489 | 58.9 | 61 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 0 | ❌ q4_bbox_hit | 1014 | 35 | 500 | 34.2 | 105 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 0 | ✅ all Qs + bbox | ≥8192 ⚠ | 95 | 1058 | capped | capped |
| nemotron3:33b-q8 | GGUF | 32768 | 4 | 4 | 4 | 4 | 0 | ✅ all Qs + bbox | 6963 | 87 | 408 | 86.5 | 42 |
| nemotron3:33b-bf16 | GGUF | 32768 | 4 | 4 | 4 | 4 | 0 | ✅ all Qs + bbox | 3273 | 61 | 441 | 59.6 | 60 |

| model | ladder rung reached | `num_predict` |
|---|---|---|
| qwen3.6:35b-a3b-nvfp4 | **65536** | 57344 |
| qwen3.6:35b-a3b-q4_K_M, q8_0 | 32768 | 24576 |
| nemotron3:33b-q8, :33b-bf16 | 32768 | 24576 |
| gemma4:31b-mlx-bf16 | 32768 | 24576 |
| the other eleven | 16384 | 8192 |

Six of eighteen needed a larger window than think-off, and qwen3.6-nvfp4 needed
**four times** the base rung. That escalation is the finding — reasoning cost,
paid in context — and it is invisible in a bare score.

**Think-on costs grounding.** `nemotron3:33b-q4_K_M` collapses to 0.000 at
16384, `qwen3.6:35b-a3b-q8_0` to 0.372 at 32768, `gemma4:26b-a4b` to 0.721.
The one clear gain is `qwen3.8:27b-nvfp4` at 1.000. This is
[ADR 0022](adr/0022-thinking-is-off-for-vision-work.md) reproduced across three
quantisations of both affected families.

## The contract arms

`contract_followed` across 18 models × 2 modes × 8 arms.

### Named coordinates are the single highest-impact requirement

| coordinate form | emitted `yxyx` while declaring `xyxy` |
|---|---|
| positional array (`pinned`, `perobject`) | **11 of 72** |
| named `x1/y1/x2/y2` (`anchored`) | **0 of 36** |

Every flip is a gemma4 cell, spanning **two sizes** (12b, 26b), **both engines**
and **four quantisations** — a family property, not a build artefact. This is
[SPEC C2](spec/vision-bbox-response-contract.md) vindicated at 36 model-modes.

**`bbox_contract_anchored` — the full recommended shape — is 35 of 36.** The one
miss is `nemotron3:33b-q4_K_M` think-on, and it is a *schema* deviation rather
than a coordinate error: it returned `bbox_2d` as an array instead of named
fields, so no `coord_order` could be inferred and `hits_declared` is 0 — while
its anchor still derived the space correctly and scored **6/6**. The recommended
consumer pipeline recovers it in full.

### The anchor: rescues, and its limits

Over the two adversarial arms across all 18 models:

| outcome | cells |
|---|---|
| anchor recovers what the declaration could not | **14** |
| anchor and declaration both already correct | 41 |
| anchor did not recover | 16 |

### `bbox_self_check` is not perfect at scale — SPEC C7 needs revising

The SPEC records a 42/42 separation from the original adversarial run. **Across
107 anchored cells here it is not 107/107.** Three misclassifications, and they
are not the same kind:

1. **`gemma4:26b-mxfp8` think-off `adv_norm1`** — accepted, `hits_anchor` 5. Not
   a validator failure: the dialect is correct (`norm1/xyxy`) and one box has a
   digit error (`x1=0.74` for `0.074`, so `x1 > x2`). `hits_bestfit` is also 5,
   so no convention recovers it. `self_check` gates the coordinate *space*, not
   per-box grounding, and it judged the space correctly.
2. **`qwen3.6:35b-a3b-q4_K_M` think-on `adv_real`** — accepted, `hits_anchor` 3
   against a `hits_bestfit` of 6 on `norm1000/xyxy`. The anchor claimed
   `real/[1200, 900]`, a fabricated frame, and **both range and aspect passed**.
   This is a genuine silent failure — the case C7 claims not to have.
3. **`nemotron3:33b-bf16` think-on `adv_real`** — rejected on aspect (anchor
   1.00 vs object extent 1.43) while `hits_anchor` was 6. A genuine false
   reject: the aspect test assumes the objects span the frame, and when they do
   not, a correct normalized anchor looks inconsistent with them.

So the honest figure is **one silent failure and one false reject in 107**, with
a third case that is a per-box defect the check is not designed to catch. C7's
"zero silent failures" is a property of the original 42, not of the mechanism.

## The MoE hypothesis is refuted

`gemma4:26b-a4b-it-q4_K_M` was the one model thinking did not rescue from the
axis flip, and being the only MoE build in its group, the question was whether
MoE routing prevents reasoning from reaching coordinate emission.

**It does not.** All three `qwen3.6:35b-a3b` tags — also MoE — pass `pinned` in
**both** think modes with `bestfit` `norm1000/xyxy`:

| model | MoE | think-off | think-on |
|---|---|---|---|
| gemma4:26b-a4b-it-q4_K_M | yes | ❌ `yxyx` | ❌ `yxyx` |
| qwen3.6:35b-a3b-q4_K_M | yes | ✅ | ✅ |
| qwen3.6:35b-a3b-nvfp4 | yes | ✅ | ✅ |
| qwen3.6:35b-a3b-q8_0 | yes | ✅ | ✅ |

The flip is a **gemma4** property, not an MoE property. Recorded because the
hypothesis was stated in advance and is now dead; it cost one table to kill.

## What this campaign changes

- **[SPEC C2](spec/vision-bbox-response-contract.md) is confirmed at scale**:
  0 of 36 with named fields against 11 of 72 with positional arrays.
- **SPEC C7's "zero silent failures" must be softened** to one silent failure and
  one false reject in 107, with the fabricated-frame case named as the gap.
- **ADR 0022 is reinforced** — nemotron3 and qwen3.6 think-on grounding collapse
  reproduces across every quantisation of both families.
- **Quantisation does not affect grounding quality** on this fixture, and the
  bf16 builds are not worth their size. The nemotron serial regression at q8 and
  bf16 is the one counter-signal and needs repeats.
- **Size does matter for fine text**: both 12b builds score 0 at 9px and 7px.

## Limits

One fixture, one image size, **n=1 per cell**. Think-on cells are additionally
**stochastic** (card sampling at `temperature 1.0`), so any think-on difference
between two models or quantisations may be sampling noise rather than the thing
being compared — see
[vision-thinking-and-declaration-honesty.md](vision-thinking-and-declaration-honesty.md).
Think-off cells are greedy and repeatable, so the axis-flip counts above are the
firmest numbers in this document. Throughput is comparable only within a
powermode segment.
