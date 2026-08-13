# Vision campaign — think-mode re-measured on-policy (2026-08-14)

Re-runs the four-model think-mode matrix of
[campaign 2026-08-13](vision-campaign-2026-08-13-thinkmode.md) with think-on cells
sampled from each model's own parameters instead of `temperature 0`.
**Short answer: "thinking hurts vision" was three different per-model facts, and one
of them was the harness.**

## Provenance

Per [ADR 0012](adr/0012-benchmark-report-templates.md). **Exploratory** report (§5):
arms are think-mode × model × replicate rather than a T1 matrix, so it is exempt from
the T1/T2 shapes but carries the provenance header and validity marks.

| | |
|---|---|
| Date | 2026-08-14 |
| Host | Apple Silicon, macOS · **powermode 0** |
| Server | `0.32.5-maxusai-31a7f1ef` · payload **b10353** · patchset 001+002+003+004+005 |
| Endpoint | `/api/generate` |
| Context | `num_ctx = 32768` |
| Output budget | `num_predict` 4 000 (think-off) / 24 000 (think-on) |
| **Sampling — think-off** | `temperature 0` (greedy) — **unchanged**, this is the control |
| **Sampling — think-on** | per model, resolved by `vision-suite/sampling.py` and recorded in each scores file as `sampling_source`: |
| | `gemma4` → `temperature 1.0, top_p 0.95, top_k 64` (card; matches what the model declares) |
| | `qwen3.6` → `temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 1.5` (card; matches what the model declares) |
| | `nemotron3` → **no overrides** — it declares none, so server defaults apply (`temperature 0.8 / top_k 40 / top_p 0.9`) |
| Replicates | **n = 3** per think-on cell (on-policy sampling is stochastic) |
| Raw data | `vision-suite/raw_onpolicy_thinkmode.json` |

## Why re-run

The 2026-08-13 campaign measured every arm at `temperature 0` and **said so in its
header** — the disclosure was not missing, and ADR 0012 §1 requires exactly that field.
The value was the problem. `temperature 0` is off-policy for thinking mode on both
families whose cards are readable, and it overrides what those models *ship*: `ollama
show` reports gemma4 declaring `temperature 1 / top_k 64 / top_p 0.95` and qwen3.6
declaring `temperature 1 / top_k 20 / top_p 0.95 / min_p 0 / presence_penalty 1.5`.
Pinning 0 discarded both.

Greedy decoding can also stop reasoning terminating at all — every token lands in
`thinking`, `eval_count` reaches `num_predict`, `response` is empty — which is what
produced that campaign's `qwen3.6 multi-image ❌ non-terminating` cell. Mechanism and
evidence: [runaway-reasoning-under-think.md](runaway-reasoning-under-think.md). Harness
fixed in `843c5705`; fallback corrected in `b4386477`.

## Results

think-off is the within-host greedy control. Per-rep values are given rather than means
wherever the **spread is the finding** — a mean would misrepresent qwen3.6 in
particular. Δ beyond ±0.01 is signal ([ADR 0012](adr/0012-benchmark-report-templates.md) §4).

### Grounding

| model | test | off | on rep 1 / 2 / 3 | verdict |
|---|---|---|---|---|
| `gemma4:26b-a4b-it-q4_K_M` | scene IoU | 0.969 | 0.967 / 0.966 / 0.967 | neutral |
| | document IoU | 0.754 | 0.759 / **0.714** / **0.707** | **mild regression** (−0.027 mean) |
| `gemma4:31b-it-q4_K_M` | scene IoU | 0.960 | 0.957 / 0.963 / 0.961 | neutral |
| | document IoU | 0.708 | 0.709 / 0.707 / **0.755** | neutral to **+0.047** |
| `qwen3.6:35b-a3b-q4_K_M` | scene IoU | 0.975 | 0.963 / **0.281** / **0.000** | **erratic** |
| | document IoU | 0.686 | 0.568 / 0.630 / 0.364 | **regressed, high spread** |
| `nemotron3:33b-q4_K_M` | scene IoU | 0.870 | **0.627 / 0.460 / 0.462** | **regressed, reproducible** |
| | document IoU | 0.045 | *(at floor — no signal)* | n/a |

**The two gemma4 lineages differ on document grounding, and only at n = 3.** From rep 1
alone the 26B looked identical to the 31B; with all three reps it regresses in two of
them (−0.040, −0.047) for a −0.027 mean, while the 31B never drops below baseline. Scene
grounding is neutral for both. This is a direct illustration of the ADR 0023 n ≥ 3 rule:
the single-run reading was wrong, and wrong in the reassuring direction.

### Validity, output stability and cost

| model | cells valid | caps | bbox dialect | document tokens (off → on) |
|---|---|---|---|---|
| `gemma4:26b-a4b` | all | 0 | stable `norm1000/xyxy` | 499 → ~3.3k (**≈6.5×**) |
| `gemma4:31b` | 12/12 | 0 | stable `norm1000/xyxy` | 499 → 2.3–3.4k (**≈5.5×**) |
| `qwen3.6:35b` | 11/12 | **1** | **flips `norm1000` → `pixel`** | 487 → 3.3–6.5k (**≈10×**) |
| `nemotron3:33b` | 12/12 | 0 | stable `norm1000/xyxy` | 467 → 6.6–18.7k (**≈25×**) |

## What changed against 2026-08-13

Cross-host, so **claims are about conclusions, not about their numbers** — see
Limitations.

| model | 2026-08-13 (temp 0, gfx1151) | on-policy (Apple) | outcome |
|---|---|---|---|
| `gemma4:26b-a4b` | document IoU **−0.092** | +0.005 / −0.040 / −0.047 | **partly reproduced**, ≈⅓ the magnitude |
| `gemma4:31b` | document IoU **−0.039** | +0.001 / −0.001 / **+0.047** | **reverses sign** |
| `qwen3.6:35b` | multi-image **non-terminating** | **terminates 3/3** | **resolved** |
| `qwen3.6:35b` | document IoU 0.000 (no change) | −0.118 / −0.056 / −0.322 | **worse than recorded** |
| `nemotron3:33b` | scene IoU **−0.449** | −0.243 / −0.410 / −0.408 | **confirmed** |

Three distinct outcomes from one harness:

**gemma4 — safe to enable, and the two lineages are not identical.** No caps and stable
dialect across all 6 think-on runs in both, and scene grounding is neutral throughout.
On document grounding they part: the **31B** is at parity or better (+0.001 / −0.001 /
+0.047), while the **26B-A4B** loses −0.027 on average (+0.005 / −0.040 / −0.047). So
2026-08-13's −0.092 for the 26B is *partly* real — about a third of it survives correct
sampling — whereas its −0.039 for the 31B does not survive at all and reverses. Thinking
is viable on both; only the 31B is free.

**qwen3.6 — the blocker moved, the problem did not.** Multi-image now converges on
every rep, so the "never terminates" finding is resolved. What remains is *instability*:
one cell still capped **on-policy**, the coordinate dialect switched from 0–1000
normalised to pixel in rep 2, and reasoning length ranged 2.3k–24k on one test. Semantic
extraction stayed correct throughout (colors 6/6 where scored), so this is output
unreliability, not comprehension loss — and it is invisible in an averaged IoU.

**nemotron3 — confirmed, and it is the one stable regression.** Scene grounding falls
to 0.627 / 0.460 / 0.462, with reps 2 and 3 reproducing to three digits and landing near
the 0.391 recorded on 2026-08-13. Every cell terminates and the dialect never moves, so
neither failure mode that sampling explains is present. Its document IoU is 0.045
think-off — the floor — so that axis carries no signal and the verdict rests on scene.

## Limitations

- **Different host and payload from the campaign this re-runs** (Apple/b10353 vs
  gfx1151-ROCm/b9888). These numbers cannot confirm or refute those directly: on the
  one directly shared cell the think-off baseline differs by **0.052**
  (gemma4:31b document 0.760 there, 0.708 here) — larger than the effect that campaign
  attributed to thinking. **The sampling fix is unconfirmed on the ROCm host** and
  should be re-run there.
- **`nemotron3` has no card-sourced sampling.** Its "on-policy" arm is *as packaged* —
  no overrides — which is the only defensible reading for a model with no published
  guidance, but it is a weaker claim than the gemma4 and qwen3.6 arms.
- **n = 3 bounds the spread, it does not estimate a rate.** qwen3.6's cap rate (1/12
  cells) and gemma4:12b's ~37.5 % upstream loop rate both need far more runs to state
  as rates.
- **`gemma4:26b-a4b`'s document regression rests on 2 of 3 reps** (−0.040, −0.047
  against +0.005). The direction is consistent but n = 3 cannot separate a real −0.027
  from a wide distribution straddling zero; more replicates would settle it. It is the
  least clean cell in this campaign.
- **Think-off is untouched and remains comparable** to every prior campaign: the
  controls reproduced their existing baselines bit-identically (gemma4:31b scene 0.960 /
  document 0.708, evals 538/499/1143/264; nemotron3 scene 0.870, evals 512/467/1049/265).

## See also

- [ADR 0023](adr/0023-think-mode-is-per-model-and-measured-on-policy.md) — the decision
  this campaign supports
- [ADR 0022](adr/0022-thinking-is-off-for-vision-work.md) — superseded; its harness traps
  and admissibility rule remain in force
- [runaway-reasoning-under-think.md](runaway-reasoning-under-think.md) — the sampling defect
- [vision-suite/sampling.py](vision-suite/sampling.py) — per-model parameters and their sources
