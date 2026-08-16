# Vision campaign 2026-08-16 — seven-model think-off sweep, with the first cross-model bbox contract matrix

The full vision suite across the current vision corpus — gemma4, qwen3.6,
qwen3.8 and nemotron3 — think-off, both engines where an MLX build exists.
This is the first sweep taken after `bbox_contract` joined the suite (#125,
#127). Earlier contract results existed for individual models, gathered ad hoc;
this is the first time **all three contract variants** have been run on **every
model under one provenance**, which is what makes the matrix below comparable
across rows.

**The sweep's quality rows are unremarkable and that is the point.** Every
model locates all six shapes, reads the serial, and extracts the invoice
perfectly. All of the signal is in the new contract columns, and it says
something the single-model results did not: the models agree almost exactly on
*where* the shapes are, and disagree constantly about *what convention they
just used*.

## Provenance

| | |
| --- | --- |
| server | `0.32.5-maxusai-a5d65906` (native macOS, no container), binary built 15:57 |
| store | `~/.ollama/models-mlx`, served on `:11436` |
| runner | `run_engine_compare.sh` with `THINK_MODES='false'` |
| power | `powermode 2` (high) for the whole sweep |
| think | `THINK=false` throughout, per [ADR 0022](adr/0022-thinking-is-off-for-vision-work.md) |
| cache | cold — daemon restarted before every model, tests back to back within a model |
| window | `num_ctx` 16384; fine-text 32768 per its own override |
| fixtures | committed `visimgs/` assets, unmodified |
| run | sweep 16:20–16:43, 0 errors; `bbox_contract_multi` repeats 18:19:25–18:25:22, 21/21 completed, same binary and power mode |

All three `nemotron3` tags in this store are GGUF; there is no MLX build, so
nemotron has no engine pair.

## Results (T1, rendered by `summarize_engine_compare.py`)

### Scene grounding + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox hits |
|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 0.960 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.958** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 16384 | 0.975 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384 | **0.965** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 0.991 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | **0.987** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 0.870 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |

### Fine-text OCR + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ all Qs + bbox | 538 | 22 | 320 | 29.4 | 123 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ all Qs + bbox | 537 | 13 | 331 | 45.2 | 80 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 2 | 2 | ✅ all Qs + bbox | 544 | 81 | 1159 | 9.0 | 401 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 1 | ✅ all Qs + bbox | 537 | 47 | 730 | 15.0 | 240 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 2 | 1 | ❌ q4_bbox_hit | 544 | 14 | 256 | 48.0 | 75 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 0 | ✅ all Qs + bbox | 547 | 18 | 370 | 37.6 | 96 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 3 | 0 | ✅ all Qs + bbox | 512 | 97 | 766 | 8.7 | 412 |

**gemma4 still owns the small tiers** — the only model in the corpus reading
the 9px tier 4/4 and the only one scoring 3 at 7px, on both engines. **qwen3.6
GGUF and nemotron own throughput**, at 401 and 412 req/h against qwen3.8 GGUF's
75. **qwen3.8 has the best scene IoU** (0.991 / 0.987) and is the only model to
hit 5 on document `name_bbox`.

Throughput rows carry the standing caveat from the
[qwen3.8 campaign](vision-campaign-2026-08-16-qwen38.md): timings on this host
move by up to 2× between identical runs. The quality rows are the citable part.

## The contract matrix

`contract_followed` — the declaration agrees with the numbers *and* loses
nothing against a best-fit search — across 7 models × 3 contract tests:

| model | engine | `bbox_contract` | `bbox_contract_multi` | `bbox_contract_reasoning` |
|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | ✅ | ❌ **yxyx** | ❌ **yxyx** |
| gemma4:31b-nvfp4 | **MLX** | ✅ | ❌ **yxyx** | ✅ |
| qwen3.6:35b-a3b-q4_K_M | GGUF | ❌ | ❌ | ❌ |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | ✅ | ❌ | ✅ |
| qwen3.8:27b-q4_K_M | GGUF | ✅ | ❌ | ✅ |
| qwen3.8:27b-nvfp4 | **MLX** | ✅ | ✅ | ✅ |
| nemotron3:33b-q4_K_M | GGUF | ❌ | ❌ | ✅ |

This is the sweep's single observation per cell. The `bbox_contract_multi`
column is superseded by the 3× repeats further down, which change one of its
rows — read the two together.

**11 of 21.** In every one of the twenty-one cells the model found all six
shapes. What varies is whether anything downstream can recover them: 11 cells
are recoverable from the declaration, 9 of the 10 failures are recoverable by a
best-fit search over type × order, and exactly one cell — qwen3.8 GGUF's
`bbox_contract_multi` — is recoverable by neither, because it declares
`norm1000` while emitting real coordinates in a frame best-fit cannot guess.
Grounding is not the variable. The declaration is.

### All seven agree on where ANCHOR is, to within a few pixels

Single-image probe, same 1920×1080 fixture. Ground truth for ANCHOR is
`[140, 160, 420, 360]`. Each model's raw box, and the same box converted back
to source pixels using **its own declaration**:

| model | declared | raw `box_2d` | → source px |
|---|---|---|---|
| gemma4 GGUF | `norm1000` | `[72, 147, 220, 335]` | `[138, 159, 422, 362]` |
| gemma4 MLX | `norm1000` | `[72, 145, 221, 336]` | `[138, 157, 424, 363]` |
| qwen3.6 GGUF | `real` [1920,1080] | `[73, 149, 221, 335]` | `[73, 149, 221, 335]` ❌ |
| qwen3.6 MLX | `norm1000` | `[73, 146, 221, 335]` | `[140, 158, 424, 362]` |
| qwen3.8 GGUF | `real` **[2500,1406]** | `[181, 212, 548, 472]` | `[139, 163, 421, 362]` |
| qwen3.8 MLX | `norm1` | `[0.073, 0.148, 0.218, 0.333]` | `[140, 160, 419, 360]` |
| nemotron GGUF | `real`, no ref | `[65, 145, 225, 333]` | — ❌ |

Read down the raw column: five of the seven wrote *the same numbers*, and one
more wrote them scaled by 1000. The models are not disagreeing about the image.
qwen3.8 GGUF looks like the outlier and is not — divide by its declared frame
and it lands within 3px, the closest row in the table after qwen3.8 MLX.

The two ❌ rows are the models whose declaration cannot convert their own
output. Same numbers as everyone else; wrong label on them.

### gemma4 transposes the axes without changing its declaration

The clearest exhibit in the sweep. Same model, same fixture, two conditions —
the numbers are *identical*, the axes are swapped, and `coord_order` reads
`xyxy` in both:

| label | `bbox_contract` | `bbox_contract_multi` |
|---|---|---|
| ANCHOR | `[72, 147, 220, 335]` | `[146, 72, 334, 221]` |
| BEACON | `[321, 108, 470, 304]` | `[107, 321, 303, 469]` |
| CIPHER | `[598, 164, 786, 391]` | `[164, 599, 390, 784]` |
| DYNAMO | `[114, 555, 251, 796]` | `[555, 114, 795, 251]` |

Best-fit resolves the right column as `norm1000/yxyx`, 6/6. Scored in the
declared dialect it is 0/6 at IoU 0.044. This is Gemma's documented `yxyx`
surfacing — but **conditionally, and silently**. It is not a fixed property to
hard-code, and it is not absent either; it is a mode the model can slip into
while still claiming `xyxy`.

### qwen3.6 GGUF has nemotron's defect, and its MLX sibling does not

qwen3.6 GGUF is 0/3, failing in the same dangerous direction as nemotron: it
declares `real` with `ref_size [1920, 1080]` and emits norm-1000. A consumer
trusting that declaration draws every box at roughly half the intended size and
half the intended offset — ANCHOR becomes 148px wide at x=73 instead of 280px
at x=140. Its MLX sibling, same weights, is 2/3 and declares `norm1000`
correctly on the single-image probe.

qwen3.6 MLX's one failure is a *hallucinated* frame rather than a wrong type:
`real` with `ref_size [1024, 768]`, an image size that was never sent.

That the two engines diverge on the same weights means the declaration is a
property of the **serving path**, not only the model — the same conclusion the
qwen3.8 GGUF `ref_size` result reached, now with a second instance.

### The `multi` column is the failure mode, not qwen3.8 — 3× repeats

The condition that breaks things is `bbox_contract_multi`: distractor images
attached, with an instruction to ignore them. The sweep gave one observation per
cell, so the column was re-run **3× per model**, one cold restart per model,
same binary and power mode:

| model | engine | `contract_followed` | how it fails |
|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | **0/3** | declares `xyxy`, emits `yxyx` |
| gemma4:31b-nvfp4 | **MLX** | **2/3** | `yxyx` on one run of three |
| qwen3.6:35b-a3b-q4_K_M | GGUF | **0/3** | declares `real` [1920,1080], emits norm-1000 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | **0/3** | declares `real` **[1024, 768]** — a frame never sent |
| qwen3.8:27b-q4_K_M | GGUF | **0/3** | declares `norm1000` [2560,1440], emits real |
| qwen3.8:27b-nvfp4 | **MLX** | **3/3** | — |
| nemotron3:33b-q4_K_M | GGUF | **0/3** | declares `real`, no `ref_size`, emits norm-1000 |

**5 of 21.** Five of the seven cells fail **3/3**, and in six of the seven the
three repeats are identical in every recorded field — same declared type, same
`ref_size`, same hit counts. This is not sampling noise; it is a stable property
of each (model, engine) pair under this prompt.

Method note: the cold restart is *per model*, so reps 2–3 run against an
already-loaded model (`load_duration` 7908ms → 276ms). `prompt_eval_count` is
unchanged across reps, so prefill is recomputed rather than served from cache,
and identical output at temperature 0 is the expected result rather than a
replay artifact — which the gemma4 MLX cell confirms by varying.

The one exception is instructive in both directions. gemma4 MLX passes 2/3,
so the axis flip is a mode it enters *sometimes* — and the sweep's single
observation of that cell happened to catch the minority outcome, which is
precisely why the repeats were run. Read the sweep's `multi` column as 1/7 and
you would conclude gemma4 MLX always flips; it does not.

The reproducer committed in #127 documents this as a qwen3.8-GGUF finding. It is
not model-specific — six of seven models fail it at least once, in four
distinct ways. Whatever the instruction to ignore attached images does, it does
to nearly the whole corpus. The mechanism remains unexplained; the rates are the
record.

### `ref_size` earns its keep

qwen3.8 GGUF's single-image row is the argument for making `ref_size` mandatory
on `real`: `hits_declared` **6/6** against `hits_bestfit` **1/6**. The
declaration — `real`, `ref_size [2500, 1406]` — is what rescues it. A best-fit
search over type × order cannot recover a frame it does not know about, and a
caller without ground truth cannot run one anyway.

## Follow-up: pinning the convention, and the placement null

The matrix above lets each model choose its convention. Three further arms were
run the same way — 7 models × 3 repeats, same distractor condition, same binary
and power mode — to find what actually drives the mis-declaration.

| arm | declaration | `contract_followed` |
|---|---|---|
| `bbox_contract_multi` | free choice, top-level | **5/21** |
| `bbox_contract_pinned` | pinned norm-1000, top-level | **21/21** |
| `bbox_contract_perobject` | pinned norm-1000, per object | **21/21** |
| `bbox_contract_anchored` | pinned, named keys, `__IMAGE__` anchor | **21/21** |

**Pinning the convention fixes it completely, and where the declaration sits is
irrelevant.** Both pinned arms return `norm1000/xyxy` at 6/6 in every cell —
nemotron3, qwen3.6 GGUF and gemma4's axis flip all included. An earlier ad-hoc
run appeared to show per-object rescuing top-level 3/3 vs 0/3; it does not
reproduce, because that prompt lacked the explicit statement of the space both
arms now share. That sentence was the active ingredient, not the placement.

The anchored arm addresses the case this corpus cannot: an image with no ground
truth. 21/21 used named `x1/y1/x2/y2` coordinates verbatim, and 21/21 returned
an `__IMAGE__` calibration entry at exactly `[0, 0, 1000, 1000]`. Compliance
with the protocol is not the weak link.

What it does **not** show: `anchor_beats_declared` is false in all 21 cells,
because under pinning nothing lied and the anchor never had to fire. Its
recovery is verified only synthetically. The decision this feeds is
[ADR 0027](adr/0027-bbox-requests-pin-norm1000-and-carry-an-anchor.md).

## Engine split

In the 21-cell matrix, MLX declares honestly in **7 of 9** cells and GGUF in
**4 of 12**. On the repeated `multi` column the split is **5 of 9** MLX against
**0 of 12** GGUF: no GGUF configuration in this corpus passes that condition,
ever, in nine attempts.

Three model families and n=1 outside the `multi` column, so this is a lead
rather than a finding. What makes it worth recording rather than dismissing is
that it lines up with two independent same-weights divergences — qwen3.6, where
GGUF fails the single-image probe and MLX passes it, and qwen3.8, whose declared
`ref_size` differs between engines on identical weights. The declaration appears
to be partly a property of the serving path.

## Corrections to previously merged docs

Three statements in already-merged docs are weaker or wrong against this sweep,
and are corrected in place:

1. [vision-bbox-coordinate-conventions.md](vision-bbox-coordinate-conventions.md)
   framed gemma4's `xyxy` as evidence that `yxyx` is a chat-template artifact.
   The order is *condition-dependent and mis-declared*, which is a worse
   problem than a fixed convention.
2. The same doc, and the reproducer's comment in `vision_suite.py`, attribute
   the `multi` failure to qwen3.8 GGUF. It is corpus-wide.
3. [vision-campaign-2026-08-16-qwen38.md](vision-campaign-2026-08-16-qwen38.md)
   says GGUF "mislocated the shape by ~250px" and calls it "genuine grounding
   error, not upscaling". That was measured wrong: the boxes are truth × 1.304
   uniformly, IoU 0.079 raw and **0.909** once divided out. A frame error, not
   a grounding error. The conventions doc already carries the corrected
   version; the campaign doc did not.

## Not covered

- Think-on. This sweep is think-off only; per-model on-policy think measurement
  ([ADR 0023](adr/0023-think-mode-is-per-model-and-measured-on-policy.md))
  exists for gemma4, qwen3.6 and qwen3.8 but not as a seven-model set.
- `nemotron3` MLX — no build in this store.
- The ROCm and CUDA lineages. Not measured; GGUF numbers here would not
  transfer.
