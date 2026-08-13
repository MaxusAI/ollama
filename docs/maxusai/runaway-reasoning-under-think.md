# Runaway reasoning under `think` — thinking that never closes

- **Status:** open finding (2026-08-13). Not a fork regression, not an engine
  defect, and **not** a structured-output defect.
- **Measured on:** server build `0.32.5-maxusai-31a7f1ef`, payload **b10353**,
  `/api/generate`, `temperature 0`.
- **Affected:** `qwen3.6:35b-a3b-q4_K_M` (GGUF, multi-image only),
  `gemma4:12b-it-q4_K_M` (GGUF) and `gemma4:12b-nvfp4` (MLX).
- **Unaffected:** `nemotron3:33b-q4_K_M` (GGUF) and `qwen3.6:35b-a3b-nvfp4` (MLX)
  converge on every suite test.

## Summary

With thinking enabled, some model × workload combinations never emit their
think-close marker. Generation ends only by exhausting `num_predict`
(`done_reason: "length"`), so every token lands in `thinking` and `response` is
empty.

**`format` plays no part.** The same cells cap with `format` omitted entirely
(evidence below). Raising `num_ctx` cannot fix it either — a larger window buys
more loop iterations, not termination.

This is a different failure from the empty-response bug that
[ADR 0002](adr/0002-deferred-format-constraining.md) /
[ADR 0004](adr/0004-routes-layer-think-format-double-request.md) fixed. There, a
grammar applied from token 0 *prevented* the marker from being emitted. Here the
reasoning pass is unconstrained exactly as those ADRs specify, and the model still
does not stop.

## Evidence 1 — `format` is not implicated

`gemma4:12b-nvfp4` (MLX), `multi_3img` prompt, `num_predict=800`, `num_ctx=16384`:

| `think` | `format` | `eval_count` | `done_reason` | `response` | `thinking` |
|---|---|---|---|---|---|
| true | `json` | 800 | length | 0 ch | 1835 ch |
| true | *(none)* | 800 | length | 0 ch | 1856 ch |
| false | `json` | 800 | length | 2066 ch | 0 ch |
| false | *(none)* | 800 | length | 2087 ch | 0 ch |

Rows 1 and 2 are the failure, and row 2 sends **no `format` at all**. Any
explanation involving a grammar — applied, deferred, or bypassed — is excluded by
row 2. Rows 3 and 4 show the same model putting tokens in `response` normally once
thinking is off. (All four stop at `length` because 800 tokens is a deliberately
small probe budget; the point is *where the tokens land*, not that they finished.)

## Evidence 2 — the mechanism is a degenerate loop

`qwen3.6:35b-a3b-q4_K_M` (GGUF), same prompt, `num_predict=3000`:

```
done_reason  = 'length'
eval_count   = 3000    (== num_predict)
thinking len = 8097 chars
response len = 0 chars
```

Tail of the reasoning:

```
- Is it possible that "hydraulic" is in Image 1? No.
- Is it possible that "quarterly" is in Image 1? No.
- Is it possible that "shipments" is in Image 1? No.
- Is it possible that "k" is in Image 1? No.
```

The model enumerates candidate strings against an image, rejects each, and never
concludes. `eval_count == num_predict` with an empty `response` is the signature;
this loop is the cause.

## Evidence 3 — escalating `num_ctx` does not converge

`run_engine_compare.sh` raises `num_ctx` whenever a cell caps. For
`qwen3.6:35b-a3b-q4_K_M` `multi_3img`:

| `num_ctx` | `num_predict` | outcome |
|---|---|---|
| 16 384 | 8 192 | capped |
| 32 768 | 24 576 | capped |
| 65 536 | 57 344 | capped |
| 98 304 | 90 112 | capped |
| 131 072 | 122 880 | **`ERROR: timed out`** — 1800 s `HTTP_TIMEOUT`, no result |

The 128 K rung yielded no measurement: at ~73 tok/s, 122 880 tokens cannot finish
inside the 30-minute request timeout. Four completed rungs, four caps.

## Evidence 4 — it is not engine-specific

Think-on matrix from the engine-compare runs. `CAP` = `eval_count` reached
`num_predict`; `ok` = valid JSON.

| model | engine | scene | document | multi_3img | finetext |
|---|---|---|---|---|---|
| `nemotron3:33b-q4_K_M` | GGUF | ok | ok | ok | ok |
| `qwen3.6:35b-a3b-nvfp4` | MLX | ok | ok | ok | ok |
| `qwen3.6:35b-a3b-q4_K_M` | GGUF | ok | ok | **CAP** | ok |
| `gemma4:12b-it-q4_K_M` | GGUF | **CAP** | **CAP** | **CAP** | ok |
| `gemma4:12b-nvfp4` | MLX | **CAP** | ok | **CAP** | **CAP** |

`gemma4:12b` caps on both engines. MLX constrains generation through the fork's own
pure-Go grammar ([ADR 0009](adr/0009-mlx-pure-go-constrained-sampling.md)) and GGUF
goes through llama.cpp, so a defect in either path cannot explain a failure present
in both. The determining factors are the **model and the workload**.

## What this is not

These caps were initially attributed to llama.cpp
[#20345](https://github.com/ggml-org/llama.cpp/issues/20345) — *"grammar is not
applied at all when thinking is enabled"* — which names Qwen3.5-35B-A3B and
Qwen3-VL-8B. **That attribution was wrong.** #20345 describes an answer emitted
unconstrained *after* thinking closes. In every case measured here thinking never
closes, `response` is empty, and the failure reproduces with no `format` at all.
The related ollama issues [#17705](https://github.com/ollama/ollama/issues/17705)
and [#17706](https://github.com/ollama/ollama/issues/17706) (both 2026-08-12)
describe the same post-thinking shape and are likewise not this.

Whether our payload is *also* subject to #20345 is untested — it would only be
observable on a cell where thinking terminates.

## Harness note

`vision_suite.py` omits the `think` field when `THINK=on`, relying on the model
default, and sets `think:false` otherwise. For `gemma4:12b-nvfp4` the omitted-field
path returns **no `thinking` key at all** while still charging `eval_count`, whereas
an explicit `think:true` surfaces the reasoning. Both cap; only the reporting
differs. Pass `think` explicitly when probing, or reasoning tokens look like they
vanished.

## Consequences for benchmarking

- **A capped cell is not a slow cell.** Do not report s/req or req/h for it — the
  number is `num_predict / tok-s`, an artefact of the budget.
  `summarize_engine_compare.py` renders these as `capped`.
- **Do not escalate `num_ctx` indefinitely.** One escalation is a fair probe for a
  genuinely tight budget; repeated caps mean a non-terminating loop.
- **Watch the timeout.** Above ~90 K `num_predict`, `HTTP_TIMEOUT=1800` expires
  before the budget does, turning a cap into `ERROR: timed out` with no data.
- **Think-off is unaffected.** All models converge with thinking off.

## Open questions

- Would a repetition penalty, or a `stop` on the loop pattern, let these cells
  finish? Untested. [ADR 0013](adr/0013-grammar-repetition-bounded-at-llama-cpp-parity.md)
  bounds grammar repetition, not reasoning repetition.
- How prompt-sensitive is it? `multi_3img` caps for qwen3.6 GGUF and for gemma4 on
  both engines, suggesting the 3-image cross-referencing prompt is a trigger — but
  `gemma4:12b` also caps on single-image `scene_single`.
- Why does `qwen3.6:35b-a3b` cap on GGUF but not MLX, when both are the same
  architecture at similar quantization? Quantization-induced drift is the obvious
  candidate and is untested.

## References

- [ADR 0002](adr/0002-deferred-format-constraining.md),
  [ADR 0004](adr/0004-routes-layer-think-format-double-request.md) — the *fixed*
  empty-response bug, distinct from this one
- [ADR 0009](adr/0009-mlx-pure-go-constrained-sampling.md) — MLX constrained sampling
- [generate-think-format-empty-response.md](generate-think-format-empty-response.md)
- [vision-suite/README.md](vision-suite/README.md) — think-mode run guidance
