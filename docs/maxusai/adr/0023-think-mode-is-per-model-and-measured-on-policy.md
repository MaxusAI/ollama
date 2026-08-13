# ADR 0023: think mode stays available for vision, is decided per model, and is only admissible when measured on-policy

- **Status:** accepted (2026-08-13). Supersedes
  [ADR 0022](0022-thinking-is-off-for-vision-work.md), whose harness traps and
  admissibility rule are carried forward unchanged.
- **Date:** 2026-08-13
- **Deciders:** MaxusAI fork maintainers
- **Related:** [runaway-reasoning-under-think.md](../runaway-reasoning-under-think.md)
  (the sampling defect and its evidence), [ADR 0005](0005-per-model-kv-cache-type.md)
  (per-model runtime configuration precedent), [ADR 0012](0012-benchmark-report-templates.md)
  (±0.01 noise floor), [ADR 0011](0011-preflight-expectations-are-versioned-code.md)
  (`min_num_predict`)

## Context

[ADR 0022](0022-thinking-is-off-for-vision-work.md) decided that vision requests are
served with `think` off, on the strength of a campaign measured at **`temperature 0`**
across every model and arm.

`temperature 0` is off-policy for thinking mode on both families whose cards we can
read. Google specifies `temperature=1.0, top_p=0.95, top_k=64` for gemma4 across *all*
use cases; Qwen specifies `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
presence_penalty=1.5` for thinking, and names `presence_penalty` as its anti-repetition
lever. Greedy decoding can therefore stop reasoning from terminating at all: every
token lands in `thinking`, `eval_count` reaches `num_predict`, and `response` is empty.
Full evidence in [runaway-reasoning-under-think.md](../runaway-reasoning-under-think.md);
the harness has since been fixed (`843c5705`) to sample think-on cells from the model
card and to record `sampling_source` in the scores.

ADR 0022's think-on arm was thus measured in a regime the model cards contraindicate.
This ADR re-measures it on-policy and reaches a **per-model** conclusion where 0022
reached a blanket one.

## Evidence

Measured 2026-08-13, Apple Silicon, server `0.32.5-maxusai-31a7f1ef`, payload
**b10353**, `/api/generate`, `num_ctx = 32768`, `num_predict = 24000`. think-off is
greedy (unchanged, and it reproduced the pre-existing baseline bit-identically);
think-on uses each model's card values. **n = 3** per think-on cell, because on-policy
sampling is stochastic — single-run cells are not a characterisation.

**`gemma4:31b-it-q4_K_M` — thinking is viable.**

| arm | scene IoU | document IoU | multi | finetext | bbox space |
|---|---|---|---|---|---|
| off (greedy) | 0.960 | 0.708 | ✅ | ✅ | `norm1000/xyxy` |
| on rep 1 | 0.957 (−0.003) | 0.709 (+0.001) | ✅ | ✅ | `norm1000/xyxy` |
| on rep 2 | 0.963 (+0.003) | 0.707 (−0.001) | ✅ | ✅ | `norm1000/xyxy` |
| on rep 3 | 0.961 (+0.001) | **0.755 (+0.047)** | ✅ | ✅ | `norm1000/xyxy` |

12/12 cells valid. No cell capped. Coordinate dialect stable in every run. Every
document delta is ≥ 0, and rep 3's +0.047 clears the ±0.01 noise floor as a genuine
*improvement*. Reasoning costs 1.7k–3.4k tokens against 0.5k–1.1k think-off (≈4×).

ADR 0022 recorded gemma4 losing **0.04–0.09 document IoU** to thinking. On-policy the
sign reverses. That figure was an artefact of greedy decoding.

**`qwen3.6:35b-a3b-q4_K_M` — thinking remains inadvisable.**

| arm | scene IoU | document IoU | multi | bbox space | scene tokens |
|---|---|---|---|---|---|
| off (greedy) | 0.975 | 0.686 | ✅ | `norm1000/xyxy` | 544 |
| on rep 1 | 0.963 (−0.012) | 0.568 (−0.118) | ✅ | `norm1000/xyxy` | 2 299 |
| on rep 2 | 0.281 (−0.694) | 0.630 (−0.056) | ✅ | **`pixel/xyxy`** | 18 540 |
| on rep 3 | **0.0 (capped)** | 0.364 (−0.322) | ✅ | none | **24 000** |

11/12 cells valid: rep 3 `scene_single` reached `num_predict` exactly with
`json_valid: false` — **on-policy**. Every quality delta in every rep is negative, and
the spread is the finding rather than the mean: the model switched bbox dialect from
0–1000 normalised to pixel in rep 2, and reasoning length ranged 2.3k–24k on one test.
Semantic extraction stayed correct throughout (colors 6/6 where scored), so this is
output instability, not comprehension loss.

On-policy sampling moved qwen from *consistent* non-termination to *occasional*
non-termination (~8% of cells here). It is a large improvement and not a cure.

**Two further levers, recorded for completeness.** `qwen3.6:35b-a3b-q8_0` converges on
`multi_3img` even at `temperature 0` (16 677 tokens) where `q4_K_M` does not, and
`gemma4:12b` recovers under its own card values with no `presence_penalty` at all.
Raw data: `../vision-suite/raw_*.json`.

## Decision

**`think` stays available for vision, in both modes. It is not prohibited, and it is
not a global default either way — it is decided per model, on on-policy measurements.**

1. **`gemma4` — permitted, and it costs nothing but tokens.** Quality-neutral to
   slightly positive on grounding, stable output dialect, terminates on every test.
   Enable it when a reasoning trace is wanted; budget ≈4× the tokens. ADR 0022's
   grounding cost for this family is retired.
2. **`qwen3.6` — think-off remains the default; enable only deliberately.** Grounding
   regresses in every measured rep, the bbox dialect is unstable, and roughly one cell
   in twelve still fails to terminate even sampled correctly. ADR 0022's practical
   verdict survives; only its stated mechanism ("non-terminating on multi-image")
   needed correcting — multi-image now converges reliably and `scene_single` is what
   fails.
3. **`nemotron3` — unchanged from ADR 0022, and untested here.** The NVIDIA card is
   gated (HTTP 401), so no card-sourced sampling exists; `sampling.py` falls back to
   greedy and warns. Its recorded grounding collapse was measured off-policy and has
   **not** been re-examined. Do not extrapolate the gemma4 result to it.

**A think-mode claim is admissible only if the run is on-policy and replicated.**
Concretely: `sampling_source` must be present and not `legacy-greedy` /
`fallback-greedy-no-card`, and n ≥ 3 for any think-on cell. A single greedy think-on
cell is not evidence about thinking — it is evidence about greedy decoding.

**ADR 0022's admissibility rule is retained verbatim:** a think-on regression claim is
not admissible without checking `eval_count` against `num_predict` first.

## Consequences

- **The think-on half of any campaign measured before `843c5705` is not comparable to
  one measured after**, because the sampling changed. Think-**off** numbers are
  unaffected and remain valid — verified by the control arm reproducing the previous
  baseline bit-identically (scene 0.960, document 0.708, evals 538/499/1143/264).
- **ADR 0022's three harness traps carry forward unchanged**, because each still fakes
  a regression and none depends on sampling:
  1. `eval_count == num_predict` exactly is budget exhaustion inside an unclosed
     thinking block, not a vision failure.
  2. Raising `NUM_PREDICT` without `NUM_CTX` converts truncation into a hard 400.
  3. `THINK` must be the literal string `"on"`; `THINK=true` silently benchmarks with
     thinking **off**.
- **A fourth trap is added:** a think-on cell measured at `temperature 0` may fail to
  terminate for reasons that have nothing to do with the model, the build, the engine
  or the quantization. Escalating `num_ctx` does not rescue it — five rungs to 128 K
  never converged — and above ~90 K `num_predict` the 1800 s `HTTP_TIMEOUT` expires
  first, converting a cap into an error with no data.
- **Bbox dialect is now a reported axis, not just a scoring detail.** The suite's
  dual-space scorer absorbs a `norm1000` → `pixel` switch into an IoU number that reads
  as a grounding collapse. Read `bbox_space` alongside `bbox_mean_iou` on think-on
  cells; a large IoU swing with intact semantic fields is a dialect flip.
- **Host scope.** Every number here is Apple Silicon / b10353. ADR 0022's are
  gfx1151/ROCm / b9888, where the same model's think-off document IoU is 0.760 against
  0.708 here — a 0.052 host difference, *larger* than the effect 0022 attributed to
  thinking. The two sets must not be pooled, and the sampling fix has **not** been
  confirmed on the ROCm host.

## Alternatives considered

- **Keep the blanket prohibition (ADR 0022 unchanged).** Rejected: it rests on
  measurements taken in a configuration two of three model cards contraindicate, and
  its gemma4 figure reverses sign when that is corrected.
- **Lift the prohibition entirely, for all families.** Rejected: qwen3.6 regresses on
  every measured rep and still fails to terminate occasionally. A blanket permission
  would be the same mistake as a blanket prohibition, in the other direction.
- **Re-run ADR 0022's campaign on the ROCm host before deciding.** Deferred, not
  rejected — that host is unreachable from here. The per-model decisions above are
  scoped to the measured host and should be reconfirmed there.
- **Set a global non-greedy default for all sampling, think-off included.** Rejected:
  it would invalidate every published baseline, preflight expectation and release
  record for no benefit, since the failure only occurs with thinking on.
