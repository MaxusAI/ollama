# Runaway reasoning under `think` — an artefact of off-policy sampling

- **Status:** resolved (2026-08-13). The cause is the benchmark harness's sampling
  configuration, not the models, the engines, the quantization, or structured output.
- **Measured on:** server build `0.32.5-maxusai-31a7f1ef`, payload **b10353**.
- **Fix:** sample think-on cells with the model card's recommended parameters. The
  suite pins `temperature: 0` and sets no `presence_penalty`.

> **This document has been wrong twice.** It first attributed the failure to
> llama.cpp grammar issue [#20345](https://github.com/ggml-org/llama.cpp/issues/20345),
> then to model-side non-termination independent of configuration (commit `f118f3c0`).
> Both are refuted below. The correction history is kept deliberately — the earlier
> claims were circulated and the matrix in the previous revision is still quoted.

## Summary

With thinking enabled, some cells never emit a think-close marker: generation ends by
exhausting `num_predict` (`done_reason: "length"`), every token lands in `thinking`,
and `response` is empty.

The trigger is the sampling configuration. The suite generates at `temperature: 0`
with no `presence_penalty`, which is off-policy for the models under test. Qwen's
model card recommends, for thinking mode, `temperature=1.0, top_p=0.95, top_k=20,
min_p=0.0, presence_penalty=1.5`, and names `presence_penalty` (range 0–2) as the
lever for reducing repetitive output. We disabled that lever and then measured the
repetition it exists to prevent.

## Evidence 1 — sampling decides the outcome

`multi_3img` (3 images), `num_predict=24000`, `num_ctx=32768`, `/api/generate`,
`think:true`, `format:"json"`. Arm A is what the suite does today; arm B is the Qwen
card's thinking-mode configuration. Arm B is stochastic, so it is run at two seeds.

| model | arm | `eval_count` | `response` | wall |
|---|---|---|---|---|
| `qwen3.6:35b-a3b-q4_K_M` | A `temperature=0` | 24 000 (cap) | **0 ch** | 341.6 s |
| `qwen3.6:35b-a3b-q4_K_M` | B seed 11 | 10 810 | 3 346 ch ✅ | 153.9 s |
| `qwen3.6:35b-a3b-q4_K_M` | B seed 22 | 8 023 | 3 123 ch ✅ | 111.0 s |
| `gemma4:12b-nvfp4` | A `temperature=0` | 24 000 (cap) | **0 ch** | 308.4 s |
| `gemma4:12b-nvfp4` | B seed 11 | 3 278 | 1 958 ch ✅ | 72.5 s |
| `gemma4:12b-nvfp4` | B seed 22 | 2 525 | 1 807 ch ✅ | 55.3 s |

4/4 converge on-policy; 2/2 run away at `temperature: 0`. The qwen cell that
survived five `num_ctx` escalations and a 30-minute timeout finishes in **154
seconds**.

**Caveat:** arm B applies *Qwen's* recommended parameters to `gemma4:12b` as well.
That establishes sampling controls the outcome; it does **not** establish gemma4's
correct settings. A proper gemma4 arm needs gemma4's own card values.

## Evidence 2 — precision is not the cause

Full precision ladders, `think:true`, `multi_3img`. Every rung terminates.

| model | engine | rung | tokens to terminate |
|---|---|---|---|
| `gemma4:31b` | MLX | nvfp4 | 2 517 |
| `gemma4:31b` | MLX | mxfp8 | 5 307 |
| `gemma4:31b` | MLX | mlx-bf16 | 2 457 |
| `nemotron3:33b` | GGUF | q4_K_M | 9 948 |
| `nemotron3:33b` | GGUF | q8 | 15 956 |
| `nemotron3:33b` | GGUF | bf16 (F16) | 23 063 |

Higher precision does not loop — it **reasons longer**. nemotron3 needs 2.3× more
tokens at bf16 than at q4_K_M, monotonically. A fixed `num_predict` therefore becomes
a tighter budget as precision rises; nemotron3 q8 and bf16 first appeared to "run
away" at `num_predict=12000` purely because that budget was below their genuine
requirement.

This matches upstream: [google/gemma-4-12B-it #41](https://huggingface.co/google/gemma-4-12B-it/discussions/41)
reports the loop at **37.5 % at F16** versus ~60 % at 4-bit MLX — present at full
precision, so quantization was never the cause there either. Note that report varied
temperature (0.0/0.7/1.0) and top_k (0/40/64/None) but, on the published evidence,
**not `presence_penalty`** — the one parameter that changes the outcome here.

## Evidence 3 — the failure is stochastic, even at `temperature: 0`

`gemma4:12b-nvfp4`, `finetext`, think-on, same rung, minutes apart. Both paths send
the *same prompt object* — [vision_suite.py:340](vision-suite/vision_suite.py:340)
imports it: `from finetext_probe import PROMPT as FINETEXT_PROMPT` — with identical
`num_predict=28672`, `num_ctx=32768`, `temperature=0`, `format:"json"`, endpoint.

| source | `eval_count` | `json_valid` |
|---|---|---|
| `ft_gemma4_12b-nvfp4_thinkon.json` | 2 761 | **true** |
| `scores_gemma4_12b-nvfp4_thinkon.json` → `finetext` | 28 672 | **false** |

A 10× divergence under greedy decoding, where output should be bit-identical.
`temperature: 0` therefore bought **no** reproducibility while inducing the failure —
which removes the main argument for keeping it.

## What this is not

- **Not llama.cpp [#20345](https://github.com/ggml-org/llama.cpp/issues/20345)**
  ("grammar is not applied when thinking is enabled"). That describes an answer
  emitted unconstrained *after* thinking closes. Here thinking never closes and the
  failure reproduces with `format` omitted entirely.
- **Not the ADR 0002/0004 empty-response bug.** That was a grammar applied from token 0
  preventing the marker. The reasoning pass here is unconstrained as those ADRs specify.
- **Not quantization or engine.** See Evidence 2; both failing models recover under
  arm B at unchanged precision and engine.
- **Not a fork regression.** No stock arm was ever run for this failure, so the fork
  is neither implicated nor cleared — but the cause is harness configuration, which is
  identical on stock.

## Corrections to the previous revision (`f118f3c0`)

Retracted or fixed after an adversarial audit:

| claim | status |
|---|---|
| "at ~73 tok/s, 122 880 tokens cannot finish in 30 min" | **wrong number.** Measured `gen_tps` is 56.5 → 36.2 min. The conclusion held on a figure that, as written (122880/73 = 28 min), refuted it. |
| "Measured on … `/api/generate`" | **wrong for the matrix.** [run_engine_compare.sh:126](vision-suite/run_engine_compare.sh:126) sets `ENDPOINT="${ENDPOINT:-chat}"`. Only the hand probes used `generate`. |
| The 5×4 think-on matrix | **withdrawn.** Cells are single observations of a stochastic process (Evidence 3), and `num_predict` varied up to 7× across rows (8 192 → 57 344) — 2.3× between the two `gemma4:12b` rows carrying its central claim — undisclosed. |
| "gemma4:12b caps on both engines" → "the determining factors are the model and the workload" | **contradicted by its own table.** For `gemma4:12b`, `document_single` capped on GGUF but passed on MLX, and `finetext` did the reverse. |
| The degenerate-loop transcript as the mechanism for all capped cells | **over-generalized.** Reasoning text was captured for one model, one test, one budget. |
| "`nemotron3:33b-q4_K_M` … converges on every suite test" | **incomplete.** It capped `multi_3img` at the first rung and converged only after escalation to a 3× budget. |
| "Think-off is unaffected. All models converge with thinking off" | **not measured for all five.** No think-off run exists for `qwen3.6:35b-a3b-nvfp4`. |
| The five-rung ladder as one automatic escalation | **two runs.** The harness stopped at 65 536 and reported NOT CONVERGED; the 96 K and 128 K rungs were a manual re-run with an overridden ladder running only `multi_3img`. |

## Remediation

1. **Sample on-policy for think-on cells.** `temperature: 0` is hardcoded in three
   places with no env knob and no `presence_penalty` anywhere:
   [vision_suite.py:37](vision-suite/vision_suite.py:37),
   [finetext_probe.py:111](vision-suite/finetext_probe.py:111),
   [preflight/probes.py:135](vision-suite/preflight/probes.py:135).
   Follow the [ADR 0005](adr/0005-per-model-kv-cache-type.md) precedent: per-model
   configuration from each model card rather than one global constant.
2. **Run n ≥ 3 per think-on cell.** On-policy sampling is stochastic by design, and
   Evidence 3 shows greedy decoding was not deterministic in practice either. A single
   capped observation cannot distinguish "loops" from "loops sometimes".
3. **Record the sampling parameters and KV cache type in the scores.**
   [ADR 0005](adr/0005-per-model-kv-cache-type.md) already requires the KV type; the
   suite does not emit it, so these runs cannot be attributed after the fact.
4. **Re-run the think-on half of the cross-family campaign.** Those cells measured a
   configuration nobody would deploy; the req/h comparison inherits the artefact.

## Open questions

- What are gemma4's own recommended thinking parameters, and does `gemma4:12b`
  converge under them rather than under Qwen's?
- Is `presence_penalty` alone sufficient, or is `temperature > 0` also required?
  Arm B moved five parameters at once and does not separate them.
- Why is greedy decoding non-deterministic here (Evidence 3)? Batching, numerical
  variation in the MLX path, and KV-cache state are the candidates; none is tested.
- Does the loop rate under on-policy sampling match the ~37.5 % upstream reports for
  `gemma4:12b`? Two seeds is too few to estimate a rate.

## References

- [ADR 0002](adr/0002-deferred-format-constraining.md),
  [ADR 0004](adr/0004-routes-layer-think-format-double-request.md) — the *fixed*
  empty-response bug, distinct from this
- [ADR 0005](adr/0005-per-model-kv-cache-type.md) — per-model runtime configuration precedent
- [ADR 0009](adr/0009-mlx-pure-go-constrained-sampling.md) — MLX constrained sampling
- [generate-think-format-empty-response.md](generate-think-format-empty-response.md)
- [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — thinking-mode sampling
- [google/gemma-4-12B-it #41](https://huggingface.co/google/gemma-4-12B-it/discussions/41) — upstream loop report
