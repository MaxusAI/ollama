# Negative result: `temperature 0.01` is not a safe substitute for card sampling in think-on

MaxusAI-fork reference. Measured 2026-08-17.

> **Status amended 2026-08-17, same day.** This was written as "an approach that
> does not work, so nobody retries it". It is now **fork policy** — see
> [ADR 0029](adr/0029-think-on-is-measured-at-a-fixed-low-temperature.md), which
> adopts `temperature 0.01` for think-on deliberately, with the cost measured
> here accepted in exchange for reproducibility at `n <= 3`.
>
> **Nothing measured below has changed, and none of it is retracted.** Read it
> as the cost sheet for that decision rather than as a prohibition: the models
> named here as losing cells still lose them, and a `no result` cell on those
> families is now *expected behaviour*, not a vision failure to be reported.

## Why it was tried

Think-on cells are measured at card-sourced sampling per
[ADR 0023](adr/0023-think-mode-is-per-model-and-measured-on-policy.md) — for
gemma4 that is `temperature 1.0`. At `n=1` per cell this makes think-on results
uninterpretable: any difference between two models, quantisations or code
revisions may be sampling noise rather than the thing under test. The proposal
was to keep the card's other parameters and drop temperature to `0.01`, on the
reasoning that the contract arms are **categorical** (does the declaration match
the boxes) where low variance beats realism.

## It does not work

Nine models, three repeats, both think modes, the eight `bbox_contract` arms.
Rendered by `vision-suite/summarize_lowtemp.py` — exploratory arm, so exempt
from the T1/T2 shapes under [ADR 0012](adr/0012-benchmark-report-templates.md)
rule 7, but reporting `(score, num_ctx, num_predict)` per rule 8. **There is no
ladder in this arm**: it was run directly rather than through
`run_engine_compare.sh`, so every cell sits at a fixed window and a missing
score means the model did not finish *at that window*, not that it failed.

| Model | Engine | think | contract_followed | mean IoU (scored cells) | no result | num_ctx | num_predict |
|---|---|---|---|---|---|---|---|
| gemma4:12b-it-q4_K_M | GGUF | false | 3/3/3 of 8 | 0.462/0.462/0.462 | 0/0/0 of 8 | 16384 | 2200 |
| gemma4:12b-it-q4_K_M | GGUF | on | 2/2/3 of 8 | 0.939/0.939/0.939 | 6/6/5 of 8 | 16384 | 8192 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | false | 3/3/3 of 8 | 0.39/0.39/0.39 | 0/0/0 of 8 | 16384 | 2200 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | on | 6/7/4 of 8 | 0.836/0.933/0.659 | 1/1/2 of 8 | 16384 | 8192 |
| gemma4:26b-mlx-bf16 | **MLX** | false | 3/3/3 of 8 | 0.392/0.392/0.392 | 0/0/0 of 8 | 16384 | 2200 |
| gemma4:26b-mlx-bf16 | **MLX** | on | 6/8/8 of 8 | 0.71/0.974/0.973 | 0/0/0 of 8 | 16384 | 8192 |
| gemma4:26b-mxfp8 | **MLX** | false | 2/3/3 of 8 | 0.307/0.39/0.39 | 0/0/0 of 8 | 16384 | 2200 |
| gemma4:26b-mxfp8 | **MLX** | on | 7/6/7 of 8 | 0.935/0.837/0.934 | 1/1/1 of 8 | 16384 | 8192 |
| gemma4:26b-nvfp4 | **MLX** | false | 3/3/3 of 8 | 0.439/0.438/0.438 | 0/0/0 of 8 | 16384 | 2200 |
| gemma4:26b-nvfp4 | **MLX** | on | 7/5/6 of 8 | 0.969/0.968/0.969 | 1/3/2 of 8 | 16384 | 8192 |
| gemma4:31b-it-q4_K_M | GGUF | false | 5/5/5 of 8 | 0.666/0.666/0.666 | 0/0/0 of 8 | 16384 | 2200 |
| gemma4:31b-it-q4_K_M | GGUF | on | 7/7/7 of 8 | 0.845/0.845/0.844 | 0/0/0 of 8 | 16384 | 8192 |
| gemma4:31b-mxfp8 | **MLX** | false | 6/4/6 of 8 | 0.801/0.538/0.78 | 0/0/0 of 8 | 16384 | 2200 |
| gemma4:31b-mxfp8 | **MLX** | on | 6/5/6 of 8 | 0.828/0.806/0.828 | 1/2/1 of 8 | 16384 | 8192 |
| gemma4:31b-nvfp4 | **MLX** | false | 5/5/5 of 8 | 0.689/0.679/0.689 | 0/0/0 of 8 | 16384 | 2200 |
| gemma4:31b-nvfp4 | **MLX** | on | 7/6/7 of 8 | 0.844/0.83/0.844 | 0/1/0 of 8 | 16384 | 8192 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | false | 4/4/4 of 8 | 0.576/0.576/0.576 | 0/0/0 of 8 | 16384 | 2200 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | on | 2/2/4 of 8 | 0.956/0.972/0.967 | 6/6/4 of 8 | 16384 | 8192 |

Three values per cell = three repeats. `no result` counts cells whose
generation never terminated at the stated window, so they carry no score;
they are excluded from the IoU mean rather than scored as zero.

**Read the `no result` column against `think=false`.** Think-off produces a
score in every cell for every model. Think-on at `temperature 0.01` loses
**6 of 8** on `gemma4:12b-it-q4_K_M` and `qwen3.6:35b-a3b-q4_K_M`, and 0–3 on
everything else.

**The budget is excluded as the cause.** At card sampling with the *same* 8192
allowance, all three models retested — `gemma4:12b-it-q4_K_M`,
`qwen3.6:35b-a3b-q4_K_M`, `gemma4:31b-it-q4_K_M` — produced a score in 8 of 8
cells in the [18-model campaign](vision-campaign-2026-08-17-eighteen-model.md).
The missing scores are the temperature.

**Where a score survives, it is good.** `gemma4:12b` think-on scores mean IoU
0.939 on the two cells that terminate; `qwen3.6` scores 0.956–0.972 on its two.
Low temperature does not degrade the answer — it prevents the answer.

This is exactly what `sampling.py`'s own header predicts: *"The lever is leaving
greedy decoding, NOT `presence_penalty` specifically."* `0.01` is greedy in all
but name, and greedy decoding is what makes reasoning fail to terminate.

## It is not model-family-specific, and there is no structural explanation

The obvious tidy story — that this is a `gemma4:12b` quirk, that build being the
only gemma4 with a separate 52.38M CLIP projector, `audio` capability and
`requires 0.30.5` — **is refuted by the very next model.**
`qwen3.6:35b-a3b-q4_K_M` runs away just as hard and shares none of that.

Nor does any other structural variable survive:

- **Not size.** `gemma4:31b-mxfp8` caps 1–2 of 8 while `gemma4:31b-it-q4_K_M`,
  same size, caps 0.
- **Not quantisation.** `q4_K_M` spans the entire range: 6 of 8 on 12b, 0 of 8
  on 31b.
- **Not engine.** MLX and GGUF both appear at 0 and both appear mid-range.

The incidence is **bimodal**: two models fail catastrophically (5–6 of 8) and
the other seven fail mildly or not at all (0–3 of 8). Nothing measured here
predicts which group a model lands in. Recorded as an open question rather than
dressed in whichever variable happens to correlate — the same discipline that
killed the MoE hypothesis in the 18-model campaign.

## What to do instead

**Use repeats at card sampling.** That is what
[ADR 0022](adr/0022-thinking-is-off-for-vision-work.md)'s own supersession did —
`n=3` on both the gemma4 reversal and the nemotron3 regression — and it is the
only method here that preserves both interpretability and termination. It costs
3× the wall-clock; lowering the temperature costs the measurement entirely.

Do **not** reach for `SAMPLING=legacy` either. That is full greedy with no card
parameters at all, and it is strictly worse than `0.01`.

## What the run did establish

**Think-off at `temperature 0` is effectively deterministic**, which retroactively
strengthens every think-off result in the 18-model campaign, all of which were
`n=1`. Contract patterns across three repeats:

| outcome | models |
| --- | --- |
| byte-identical across all three repeats | **7 of 9** |
| varied in one arm of one repeat | `gemma4:26b-mxfp8`, `gemma4:31b-mxfp8` |

So the axis-flip counts behind [SPEC C2](spec/vision-bbox-response-contract.md)
rest on repeatable measurements, not lucky draws.

**Both models that varied are `mxfp8`**, and no other quantisation varied at all.
That is consistent with the known result that temperature 0 is not reproducible
across model loads on MLX — greedy decoding fixes only the argmax, and near-ties
still flip when a reload changes reduction order — but it is two models at n=3
and should not be promoted to a claim about mxfp8 without more.

## Provenance and limits

Server `0.32.5-maxusai-a5d65906`, cold restart per model, powermode 2,
`TEMPERATURE=0.01` recorded in the scores as
`card:<fam>+override(temperature=0.01)` — a label that only distinguishes
overridden from on-policy runs because of the fix in #140; before it, these
files would have been archived as `card:gemma4`.

- `TEMPERATURE` does not reach think-off cells at all: `sampling_for` returns
  `GREEDY` before the override loop when think is off, deliberately, since every
  published think-off baseline depends on `temperature 0`. The think-off half of
  this run is therefore a **determinism check at temperature 0**, not a
  low-temperature arm.
- One fixture, `n=3`, eight contract arms only.
- **`gemma4:31b-mlx-bf16` was deferred** for memory headroom (63.5 GB build). It
  is the most informative missing cell: `26b-mlx-bf16` is the one 26b build that
  never caps, so a clean 31b bf16 would be the strongest available evidence for
  a quantisation reading — which the rest of this table currently contradicts.

### A harness trap worth not repeating

The first attempt ran `vision_suite.py` directly and inherited the **think-off
default of `num_predict` 2200** for think-on cells, so 7 of 8 gemma4:12b cells
capped at 2200. That is
[ADR 0022](adr/0022-thinking-is-off-for-vision-work.md)'s harness trap #1 —
budget exhaustion "recurs at *every* budget below the model's real thinking
length" — and it confounds precisely this experiment, because a cap from too
small a budget is indistinguishable from a cap from near-greedy decoding. Those
outputs were discarded rather than scored. `run_engine_compare.sh` derives the
think-on budget automatically; a hand-run probe does not.
