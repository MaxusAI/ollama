# Thinking and declaration honesty: a second axis ADR 0022 does not cover

MaxusAI-fork reference. Written 2026-08-16 from the 18-model × both-think-modes
full-suite run. **Interim: the run was still in progress when this was written,
n=1 per cell.** The direction is consistent across every model measured so far;
the rates will be restated when it completes.

## The claim

[ADR 0022](adr/0022-thinking-is-off-for-vision-work.md) concluded that thinking
is off for vision work, and [ADR 0023](adr/0023-think-mode-is-per-model-and-measured-on-policy.md)
required it to be measured per-model and on-policy. Both are about **grounding
quality** — IoU, extraction recall, whether the shape is found.

**Declaration honesty is a different axis, and it moves the other way.** Whether
a model correctly describes the coordinate convention it just used is not the
same question as whether it located the shape, and the evidence so far is that
thinking *helps* the former while ADR 0022 shows it hurting the latter.

This matters because the two verdicts point in opposite directions for the same
request, and nothing in the ADRs currently says so.

## Read this before the table

**The two columns are not measured the same way, and the difference could
account for the entire result.** Think-off runs greedy (`temperature 0`,
deterministic). Think-on runs card-sourced sampling per
[ADR 0023](adr/0023-think-mode-is-per-model-and-measured-on-policy.md) — for
gemma4 that is `temperature 1.0, top_p 0.95, top_k 64`, which is **stochastic**.

So every ❌ below is a deterministic, repeatable failure, and every ✅ in the
think-on column is **one draw from a sampler**. The comparison is a certain
outcome against a single sample.

Taken at face value the direction looks real: think-off flips in 5 of 5 flipping
models, think-on in 1 of 5. But if think-on's true flip rate were 50%, observing
1 flip in 5 single draws has p ≈ 0.16 — not close to sufficient. **This
document does not establish that thinking rescues the axis flip.** It
establishes that the flip is deterministic without thinking and *not* determined
with it, which is a much weaker statement.

The confound is general, not specific to this finding: with `temperature 1.0`
and n=1, any think-on difference between two models, quantisations or code
revisions may be sampling noise rather than the thing being compared. Every
think-on cell in this run carries it.

**What would settle it**, in ascending cost: re-run the six contract arms at
`TEMPERATURE=0.01` (keeps the card's `top_k`/`top_p` and, for qwen3.6, the
`presence_penalty 1.5` anti-repetition lever, so it is *not* the greedy
configuration that broke ADR 0022) — appropriate because the contract arms are
**categorical**, where low variance matters more than realism; or run think-on
at card sampling with 3+ repeats, which is what ADR 0022's own supersession did
(`n=3` on both the gemma4 reversal and the nemotron3 regression).

## Measured

`bbox_contract_pinned` and `bbox_contract_perobject` — pinned norm-1000,
positional array, distractor condition — scored by whether the declaration
matches the boxes:

| model | engine | think-off | think-on |
| --- | --- | --- | --- |
| gemma4:12b-it-q4_K_M | GGUF | pinned ❌ (`yxyx`) | ✅ |
| gemma4:12b-nvfp4 | MLX | ✅ | ✅ |
| gemma4:26b-a4b-it-q4_K_M | GGUF | ❌ ❌ (`yxyx`) | **❌ ❌ (`yxyx`)** |
| gemma4:26b-nvfp4 | MLX | ❌ ❌ (`yxyx`) | ✅ ✅ |
| gemma4:26b-mxfp8 | MLX | ❌ ❌ (`yxyx`) | ✅ ✅ |
| gemma4:26b-mlx-bf16 | MLX | ❌ ❌ (`yxyx`) | ✅ ✅ |
| gemma4:31b-it-q4_K_M | GGUF | ✅ ✅ | *(pending)* |

Every failure is the same failure — the axis transposition of
[SPEC C2](spec/vision-bbox-response-contract.md) — and in every one the model
**found all six shapes**: `hits_bestfit` is 6/6 on `norm1000/yxyx` throughout.
Thinking is not improving vision here. It is improving the model's account of
what it did.

## The exception is the interesting part

**`gemma4:26b-a4b-it-q4_K_M` flips in both modes.** It is the only MoE build in
the 26b group, and it is the only one thinking does not rescue. Three dense 26b
variants — nvfp4, mxfp8, bf16, spanning three quantisations and one engine — all
stop flipping with think-on; the MoE one does not.

That is a single model at n=1, and it is the kind of observation this fork has
been wrong about three times already this session by narrating from one cell.
Recorded as a question, not a finding: **does the MoE routing change whether
reasoning reaches the coordinate-emission step?** `qwen3.6:35b-a3b` is also MoE
and is still pending in this run — if it flips in both modes too, that is worth
pursuing; if it does not, this is noise.

## What this does not license

**It is not an argument for turning thinking on for vision serving.** ADR 0022's
grounding verdict stands on its own measurements, and this run has not yet
re-measured IoU across both modes. The two axes can both be true: thinking can
cost grounding precision *and* improve self-description at the same time, and a
serving decision has to weigh which one the consumer depends on.

**It is also not a reason to relax C2.** The cheaper fix for every cell in the
table above is named coordinates, which cost nothing, work in both modes, and
are **0 of 13** on this failure — against **11 of 26** for positional arrays.
Thinking rescues most of the flips at roughly triple the wall-clock; naming the
fields rescues all of them for free.

The practical reading: **use named coordinates, and do not rely on thinking to
save a positional-array schema.** If a schema is stuck with positional arrays
for compatibility reasons, think-on is a partial mitigation with a known
exception, not a fix.

## Provenance

Server `0.32.5-maxusai-a5d65906`, cold restart per model,
`THINK_MODES='false on'`, 12 tests per model-mode, `num_ctx` 16384,
`num_predict` 2200 think-off and 8192 think-on.

**Sampling** (ADR 0005 requires this recorded; an earlier draft of this document
omitted it, which is the omission that let the confound above go unstated):

| mode | `sampling_source` | resolved |
| --- | --- | --- |
| think-off | `greedy-think-off` | `temperature 0` — deterministic |
| think-on, gemma4 | `card:gemma4` | `temperature 1.0, top_p 0.95, top_k 64` — **stochastic** |
| think-on, qwen3.6 | `card:qwen3.6` | `temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 1.5` |
| think-on, qwen3.8 / nemotron3 | `packaged-defaults-no-card` | no overrides sent; card gated or absent |

Known defect in the recording itself: `provenance()` derives `sampling_source`
from the model family alone, so a run using the `TEMPERATURE`/`TOP_P`/… env
overrides is still labelled `card:<fam>` while not carrying card values. The
resolved `sampling` dict does show the override, so it is recoverable, but the
source label is wrong. Fix that before running any low-temperature arm, or the
archive cannot distinguish the two.

**Power mode is not constant across this run** and is stamped per model-mode in
the runner log. Three segments: powermode 2 for models 1–2, powermode 1 for
models 3–6, powermode 2 from model 7. Quality metrics are power-invariant at
temperature 0, so every number above is unaffected; **throughput rows from this
run are only comparable within a segment** and are not cited here.

One capped cell so far — `gemma4:12b-it-q4_K_M` think-on `finetext`, with
`eval_count == num_predict == 8192`. Per ADR 0022's harness trap that is budget
exhaustion inside an unclosed thinking block, **not** a vision failure, and it is
excluded rather than scored.
