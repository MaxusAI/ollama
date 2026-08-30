# mlx-cuda vs cuda decode throughput — 2026-08-30

The README claimed the MLX runtime ran at "roughly half the throughput of the
GGML path" on CUDA, and said in the same breath that the figure was an
operational observation with **no matched same-host, same-model pair recorded**.
This records those pairs.

- Harness: [`vision-suite/bench_engine_throughput.py`](vision-suite/bench_engine_throughput.py)
- Raw: [`vision-suite/bench-runs/mlx-cuda-vs-cuda-0.33.2.json`](vision-suite/bench-runs/mlx-cuda-vs-cuda-0.33.2.json)
- Superseded first attempt, kept deliberately:
  [`…-pass1-short-window.json`](vision-suite/bench-runs/mlx-cuda-vs-cuda-0.33.2-pass1-short-window.json)

**Scope: `mlx-cuda` and `cuda` only.** Nothing here applies to `mlx-metal`. The
MLX arm is nvfp4, an NVIDIA FP4 format; a Metal MLX arm would be a different
quantization on different hardware, so the two platforms are not measuring
comparable things and neither result may be carried to the other.

## Setup

`0.33.2-dynres-5-g2b95b4a`, one container, one server process — SPEC H11 makes
`server_version` the comparability boundary, and a single process guarantees it
rather than asserting it. GPU 0 (RTX PRO 6000 Blackwell) pinned; the host's
second card is an 11 GiB 2080 Ti and a run split across the two would measure
the split. `num_ctx=8192` pinned, `num_predict=256`, `temperature=0`,
`think=false`, `OLLAMA_MAX_LOADED_MODELS=1`. Every sample terminated at
`done_reason: "length"`, so `eval_count` is a constant 256 across arms and decode
rate is the only free variable. n=12 after 3 discards; arms interleaved; the
headline pair repeated last as an order control.

Pairs verified matched via `/api/show` — same `block_count`, `embedding_length`
and parameter count on both sides — before any of it was measured.

## Result

Estimator is the tail-half median, with the first-half/second-half shift beside
it. An arm with a large shift has no single throughput to quote, and is marked.

| pair | arch | `mlx-cuda` | half-shift | `cuda` | half-shift | ratio |
|---|---|---|---|---|---|---|
| `gemma4-31b` | dense | 47.01 | +1.8% | 62.67 | +0.1% | **75.0%** |
| `qwen3.8-27b` | dense | 36.56 | +7.4% ⚠ | 69.49 | −0.5% | ≥52.6% |
| `gemma4-26b` | MoE 8/128 | 65.72 | −0.8% | 191.11 | −0.3% | **34.4%** |
| `qwen3.6-35b-a3b` | MoE 8/256 | 38.80 | +23.2% ⚠ | 98.86 | −1.5% | ≥39.2% |

**`mlx-cuda` runs at 34–75% of `cuda`, median 46%.** The two ⚠ arms were still
climbing in their tails, so those ratios are lower bounds — those models do not
have a fixed throughput to quote.

"Roughly half" survives as a *central* estimate and fails as a description of any
individual case: the spread is 2.2×, and it is not architectural — the two dense
models sit at 75% and 53%, the two MoE at 34% and 39%.

`gemma4:26b-a4b-it-q4_K_M` decoding at 191 tok/s is not an anomaly:
`expert_used_count=8` of 128 at `expert_feed_forward_length=704` is roughly 4B
active parameters per token, so a 26B MoE outrunning the dense 31B threefold is
expected.

## The more actionable finding: variance

Intra-arm spread as a percentage of the arm's own median, all 20 arms across both
runs:

```
cuda       1.1  1.7  2.2  2.3  3.0  3.1  4.3  5.2  5.4  6.0
mlx-cuda   3.4  7.3  8.3 10.6 13.6 15.6 27.3 29.7 36.0 45.9
```

`mlx-cuda`'s per-request variability is **~5× `cuda`'s at the median and reaches
46% within a single arm**, same host, same window, same prompt, back to back. For
anything that sets a timeout, promises a latency, or compares two builds, that
matters more than the ratio — and no single-number claim can express it.

It is per-model, not a blanket property: `gemma4:31b-nvfp4` measured 45.97 /
46.46 / 46.19 / 46.44 across four independent measurements spanning two container
lifetimes.

## Why the first run is kept

The first attempt used `discard=1, n=5` and would have published **28.7%, 37.9%,
41.2%, 73.2%** — three of four wrong, every one of them low. The sampling error
hit `mlx-cuda` and never `cuda`, so the errors all pointed the same way and the
result read as a clean confirmation of the claim under test. Errors that all
point one way are worse than noisy ones.

The order control is what caught it: `gemma4-31b` re-measured at the end of the
run agreed with its own opening figure to 1.1%, which ruled out whole-run drift
and localised the problem to the `mlx-cuda` arms specifically.

## Stationarity is evidence, not an assumption

The `cuda` arm was flat first time on all ten arms (drift within ±1.5%). The
`mlx-cuda` arm is often not stationary, and the profile is per-model:

| model | profile |
|---|---|
| `gemma4:31b-nvfp4` | cold 1.42 tok/s → full speed (47) on request 2, flat after |
| `gemma4:26b-nvfp4` | cold 41.99 → climbs over 3 requests → real plateau |
| `qwen3.8:27b-nvfp4` | cold 2.31 → plateau ~25 → **humps** to 38.6 at request 8 → decays to 34 by request 15 |
| `qwen3.6:35b-a3b-nvfp4` | settled instantly at n=5 in one run; needed 12 requests to reach the same value in the next |

Even the cold request differs 30× between models (1.42 vs 41.99). Three
heuristics were tried and each falsified by the data:

- *"MLX needs a longer warm-up"* — two arms reach full speed on request 2.
- *"Discard 3 and you are safe"* — one arm needed 12. A discard count is a
  floor, not evidence.
- *"A flat drift figure means settled"* — the same arm gave +0.5% at n=5 and
  +23.2% at n=12. First-to-last drift is noise-dominated; it reported −8.6% on
  an arm with no trend at all.

What survived: the **tail-half median**, reported with the half-shift, and the
fact that tails agree across independent runs even when the approach to them does
not.

## Limits

- **One host, one GPU, one build.** The ratio will move with GPU and model.
- **Engine and quantization move together** (nvfp4 vs q4_K_M). This measures the
  two stacks as shipped, not the engine in isolation — the same confound the
  README already states for output quality.
- **Decode only.** `prompt_eval_duration` is not trustworthy on the `mlx-cuda`
  arm: the same arm reported 369 tok/s on one request and 3,348,513 tok/s on the
  next, while `cuda` reported coherent figures throughout. Prefill numbers are
  recorded in the raw JSON but must not be quoted without re-establishing that.
- **The `qwen3.8-27b` hump is unexplained.** A rise-then-decay under sustained
  load has the shape of a boost-clock profile, but the `cuda` arm on the same
  model in the same session was flat to ±0.2%, which no purely thermal account
  explains. Recorded as an open anomaly rather than given a mechanism.
- **n=1 host.** Two runs agree, but both are this machine.

## Incidental

The one-time MLX JIT on a fresh container measured **437.6 s and 443.9 s** on two
independent starts of 0.33.2 — consistently below the 590–660 s previously
recorded. Not a hang; size timeouts accordingly.
