# Vision campaign — think-mode at 64K context (2026-08-13)

Does enabling `think` help vision work? Measured across four models, both modes, on the
gfx1151 host. **Short answer: no — and on two of the four it is actively harmful.**

## Provenance

Per [ADR 0012](adr/0012-benchmark-report-templates.md). This is an **exploratory** report
(§5): the arms are free-form think-on/off × model × test rather than a T1 campaign matrix,
so it is exempt from the T1/T2 shapes but carries the provenance header and validity marks.

| | |
|---|---|
| Date | 2026-08-13 |
| Host | Ryzen AI Max+ 395 / Radeon 8060S (**gfx1151**), 96 GiB VRAM, Linux, ROCm |
| Power mode | `n/a` (Linux) |
| Server | `0.32.1-dynres-296eb020` (`v0.32.1-dynres.3`) · payload **b9888** · patchset 001+002+004+005 |
| Endpoint | `/api/chat`, temperature 0 |
| Context | **`num_ctx = 65536`** (128K for the qwen escalation) |
| Output budget | `num_predict` 12,000 → 64,000 depending on arm (see below) |
| Serving | `OLLAMA_NUM_PARALLEL=1`, bench container on `:11437`, model store read-only |
| Harness | `vision-suite` from `fix/vision-suite-thinkmodes-and-finetext-source` (PR #85) |

## Results

think=off figures are the `v0.32.1-dynres.3` release-record run; think=on measured here.
`—` = not measured. Δ beyond ±0.01 is signal (ADR 0012 §4 noise floor).

| model | test | off | on | Δ | eval tokens (on) |
|---|---|---|---|---|---|
| **gemma4:26b-a4b-it-q4_K_M** | scene IoU | 0.973 | 0.964 | −0.009 *(noise)* | 4,137 |
| | document IoU | 0.810 | **0.718** | **−0.092** | 3,140 |
| | document items | 5/5 | 5/5 | — | — |
| | multi-image | ✅ all | ✅ **all** | — | 6,021 |
| **gemma4:31b-it-q4_K_M** | scene IoU | 0.961 | 0.961 | **0.000** | 2,218 |
| | document IoU | 0.760 | **0.721** | **−0.039** | 1,953 |
| | document items | 5/5 | 5/5 | — | — |
| | multi-image | ✅ all | ✅ **all** | — | 2,641 |
| **qwen3.6:35b-a3b-q4_k_m** | scene IoU | 0.953 | 0.962 | +0.009 *(noise)* | 19,160 |
| | document IoU | 0.320 | 0.320 | 0.000 | 3,139 |
| | fine text | valid | valid | — | 1,668 |
| | multi-image | ✅ all | ❌ **never terminates** | — | **>64,000** |
| **nemotron3:33b-q4_K_M** | scene IoU | 0.840 | **0.391** | **−0.449** | 11,749 |
| | document IoU | 0.061 | 0.017 | −0.044 | 8,462 |
| | document items | 5/5 | **4/5** | **−1** | — |
| | multi-image | ✅ all | — | — | — |

## Findings

**1. Thinking never improves grounding; it degrades it or does nothing.** The only positive
deltas (qwen scene +0.009, gemma4:26b scene −0.009) sit inside the ±0.01 noise floor.
Every movement that clears the floor is **negative**: gemma4:26b document −0.092,
gemma4:31b document −0.039, nemotron3 scene −0.449.

**2. nemotron3 is harmed badly.** Scene grounding more than halves (0.840 → 0.391) and it
*loses* an extraction item (5/5 → 4/5) while labels, serial, invoice number and total stay
correct. Comprehension is intact; **spatial precision is not**. Plausibly it reasons its way
off coordinates it would otherwise read directly.

**3. qwen3.6 does not terminate on multi-image cross-referencing.** Budgets of 12,000,
32,000 and 64,000 tokens were each consumed exactly, with no partial convergence:

| num_predict | num_ctx | eval_count | outcome |
|---|---|---|---|
| 12,000 | 65,536 | 12,000 | exhausted |
| 32,000 | 65,536 | 32,000 | exhausted |
| 64,000 | 131,072 | 64,000 | exhausted |

It is a **loop, not slow convergence**. Captured thinking on the suite's `MULTI_PROMPT`
shows a unique/total token ratio of **0.211** with 15-grams repeating ×3 — the model
re-enumerates the same cross-image word list indefinitely. The prompt asks it to be
"exhaustive" about words shared across three images, and it never closes that check. The
same model on a *generic* three-image prompt finishes in **2,113 tokens** with coherent,
non-repeating reasoning, so this is prompt-specific, not a model or build defect.

The hard ceiling is `262,144 − 6,134 = 256,010`, so there is no budget that rescues this.

**4. gemma4 is the only family where think-on is viable.** Both models terminate in
2.0k–6.0k tokens on every test — roughly 10× shorter than qwen3.6 — and both pass
multi-image, which qwen3.6 cannot. But viable is not beneficial: document grounding still
degrades on both.

## Guidance

| model | think-on for vision? |
|---|---|
| `nemotron3` | **No.** Grounding collapses and extraction loses an item. |
| `qwen3.6` | **No.** Non-terminating on multi-image; elsewhere +0.009 IoU for 19k tokens. |
| `gemma4` (26b, 31b) | **Safe, but pointless.** Only family that completes everything. Costs 0.04–0.09 document IoU and buys nothing measurable. Enable only if you want the reasoning trace. |

Cost matters independently of quality: qwen3.6's 19,160-token scene answer is ~5.5 min at
57 tok/s against ~15 s with thinking off.

## Two harness traps — both look exactly like vision failures

Recorded because either one, taken at face value, produces a false regression report.

**Budget exhaustion masquerading as failure.** The first run returned `json_valid=false`
and `bbox_mean_iou=0.0` on all three vision tests, which reads as "think-on catastrophically
breaks vision on this build". It does not. The tell is `eval_count` equal to `num_predict`
**exactly** — the model spent its whole allowance inside an unclosed thinking block and never
emitted JSON. `preflight/expectations.toml` documents this at the ~600-token floor
(`min_num_predict`); it recurs at *every* budget below the model's actual thinking length.
**Always check `eval_count` against `num_predict` before believing a think-on failure.**

**Raising the budget alone converts truncation into a hard 400.** The server checks
`prompt + num_predict <= num_ctx` up front, so `NUM_PREDICT` must rise *with* `NUM_CTX` —
both runners now derive `num_predict` from the rung as `num_ctx - CTX_PROMPT_RESERVE`
(`6c90d7bb`, reserve corrected to 8192 in `561e6b98` once nemotron3's 6,203-token
multi-image prompt overran the original 4096).

**`THINK` must be the literal string `on`.** `vision_suite.py` and `finetext_probe.py` both
test `== "on"`, and `run_engine_compare.sh` defaults it to `false`. Passing `THINK=true`
silently runs with thinking **off** — a "thinking benchmark" that isn't one.

## Limitations

- nemotron3 multi-image with think-on was not measured; the two arms that were measured both
  regressed, so it was not pursued.
- Fine-text was measured for qwen3.6 only.
- Single run per cell. Quality cells are bit-reproducible at temperature 0 per
  (payload, backend, budget, image), so repeats were not taken; the non-termination result
  is from three independent escalations rather than one.
- gemma4's think-on figures come from single-test invocations, not a full
  `run_engine_compare.sh` pass, so no `s/req` or `req/h` pair is reported for them —
  throughput here would be dominated by thinking length and is not comparable to the
  release-record cells.
