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

## Results — think-off (T1, rendered by `summarize_engine_compare.py`)

### Scene grounding + document extraction

| Model | Engine | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice | name_bbox |
|---|---|---|---|---|---|---|
| gemma4:12b-it-q4_K_M | GGUF | 0.870 | 6/6 · 6/6 · **5/6** | ✅ | 5/5 · 5/5 · ✅ | 3 |
| gemma4:12b-nvfp4 | MLX | 0.953 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 0.969 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-nvfp4 | MLX | 0.965 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-mxfp8 | MLX | 0.971 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-mlx-bf16 | MLX | 0.973 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-it-q4_K_M | GGUF | 0.960 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-nvfp4 | MLX | 0.958 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-mxfp8 | MLX | 0.960 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-mlx-bf16 | MLX | 0.967 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 0.975 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.6:35b-a3b-nvfp4 | MLX | 0.965 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.6:35b-a3b-q8_0 | GGUF | 0.967 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| **qwen3.8:27b-q4_K_M** | GGUF | **0.991** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | **5** |
| **qwen3.8:27b-nvfp4** | MLX | **0.987** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | **5** |
| nemotron3:33b-q4_K_M | GGUF | 0.870 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q8 | GGUF | 0.884 | 6/6 · 6/6 · 6/6 | **❌** | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-bf16 | GGUF | 0.870 | 6/6 · 6/6 · 6/6 | **❌** | 5/5 · 5/5 · ✅ | 4 |

Grounding is essentially saturated: sixteen of eighteen exceed 0.95, and every
model finds all six shapes. **Quantisation barely moves quality.** gemma4:31b
spans 0.958–0.967 across q4_K_M, nvfp4, mxfp8 and bf16 — inside ADR 0012's
±0.01 noise floor. The gemma4:26b family spans 0.965–0.973. Paying 3.4× the
disk for bf16 over q4_K_M buys nothing measurable here.

**The two nemotron serial failures are new** and are a quantisation effect in
the *opposite* direction to the intuition: `q4_K_M` reads the serial correctly
while `q8` and `bf16` do not. That is one fixture at n=1 and should be repeated
before anyone acts on it.

### Fine text, multi-image, throughput

The **12b tier is where fine text breaks down**: both 12b builds score **0** at
the 9px and 7px tiers, where every 26b and 31b build scores 3–4. That is the
clearest size effect in the campaign and it is not subtle.

`gemma4:26b-mxfp8` at 339 req/h and `qwen3.6:35b-a3b-nvfp4` at 485 req/h lead
throughput; `gemma4:31b-mlx-bf16` is last at 44 req/h. Read those only against
other powermode-2 rows.

## Results — think-on

**Think-on is a grounding disaster for nemotron3 and qwen3.6, and this is the
strongest evidence yet for [ADR 0022](adr/0022-thinking-is-off-for-vision-work.md).**

| Model | scene IoU off | scene IoU on | note |
|---|---|---|---|
| nemotron3:33b-q4_K_M | 0.870 | **0.000** | capped — `eval_count == num_predict` |
| nemotron3:33b-bf16 | 0.870 | **0.000** | capped |
| nemotron3:33b-q8 | 0.884 | **0.234** | ladder climbed to 32768 |
| qwen3.6:35b-a3b-q8_0 | 0.967 | **0.372** | ladder climbed to 32768 |
| qwen3.6:35b-a3b-q4_K_M | 0.975 | **0.835** | ladder climbed to 32768 |
| gemma4:26b-a4b-it-q4_K_M | 0.969 | **0.721** | |
| gemma4:31b-mlx-bf16 | 0.967 | **0.797** | ladder climbed to 32768 |
| qwen3.8:27b-nvfp4 | 0.987 | **1.000** | the one clear gain |

The `0.000` cells are **capped, not scored**: `eval_count == num_predict`, which
per ADR 0022's harness trap is budget exhaustion inside an unclosed thinking
block, not a vision failure. Seven cells across the campaign are capped, all of
them nemotron3 think-on. The `num_ctx` ladder climbed on eight model-modes —
every qwen3.6 tag, every nemotron3 tag, and `gemma4:31b-mlx-bf16` — which is the
non-termination ADR 0022 documented, now reproduced across three quantisations
of both families.

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
