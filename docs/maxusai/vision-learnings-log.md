# Vision harness — learnings log

Append-only. Each entry is a **claim sharp enough to be wrong**, the number that
supports it, and the artefact that now enforces it. An entry with no enforcing
artefact is a note, not a learning — it will be forgotten and re-learned.

Format: `### YYYY-MM-DD — claim` / **Evidence** / **Enforced by** / **Cost**.
A claim that later fails is **kept and marked REFUTED**, never deleted — the
sequence claim → retraction → refutation is more instructive than the final
answer, and a deleted claim can be re-derived by the next person.
"Cost" is what not knowing it already cost, in cells, hours, or wrong published
numbers. It is the field that decides whether the enforcement was worth building.

---

## 2026-08-19 — model behaviour

### 2026-08-19 — Pinning norm-1000 removes image geometry as a variable
A model that never has to *name* the frame it works in cannot leak its internal
resize into the coordinates.

- **Evidence** — norm-1000 pin: **55 of 56** cells convert 6/6 across 14
  geometries × 2 models × 2 think modes. Same models, same geometries, `real`
  pin: qwen3.8 **14/14**, qwen3.6 **1/14**.
- **Enforced by** — SPEC C13–C18 §4, [ADR 0030](adr/0030-bbox-conformance-is-scoped-to-image-geometry.md).
- **Cost** — the contract shipped on a declaration-honesty argument (21/21 vs
  5/21) when a correctness argument was available and stronger.

### 2026-08-19 — The reported frame is not a function of input size
It snaps to canonical sizes and moves in **both** directions.

- **Evidence** — qwen3.8 returns `2560×1440` for **4 different inputs** and
  `2337×1754` for **2 more**. Frame/input ratio spans **0.67×–1.44×** (qwen3.8)
  and **0.46×–2.50×** (qwen3.6). 8 of 13 qwen3.6 cells and 6 of 13 qwen3.8 cells
  are below 1.0×.
- **Enforced by** — SPEC C17 (amended; its original text asserted the opposite),
  `bbox_contract_real_1img`.
- **Cost** — C17 shipped claiming every usable anchor reports a frame *larger*
  than the input. That was an artefact of only ever sending 1920×1080.

### 2026-08-19 — qwen3.6 think-on does not terminate; the anchor bounds it
Not an efficiency gap — a qualitative difference.

- **Evidence** — `multi_3img`: **122,880 tokens** (= `num_predict`), **313,054
  characters** of reasoning, **0 characters** of answer, byte-identical across 2
  reps. `multi_3img_anchored`: **10,910 tokens**, complete and correct.
  Reproduces on Apple Metal *and* CUDA Blackwell, two different builds.
- **Enforced by** — [campaign doc](vision-campaign-2026-08-19-qwen36-anchored.md),
  `CTX_LADDER` 131072 rung.
- **Cost** — the unanchored configuration has no req/hour rate at all. It was
  previously recorded as a `q4_bbox_hit` ❌, which reads as a grounding failure.

### 2026-08-19 — C7's silent-failure rate is a property of the image, not the validator
- **Evidence** — **4 silent failures in 14** qwen3.6 cells under a `real` pin
  (`self_check` passes, anchor converts 2–3 of 6), against the **1 in 107**
  measured at 16:9 HD. 9 of 14 correctly rejected.
- **Enforced by** — SPEC C14 (response-only conditions), C7 amendment.
- **Cost** — a single-number failure rate was quoted for a validator whose rate
  varies by two orders of magnitude with the input.

### 2026-08-19 — The anchor's value is per-model and inverts
- **Evidence** — under a `real` pin, qwen3.8's anchor converts **6/6 at all 14**
  geometries while best-fit reaches only 1/6 in five of them. qwen3.6's anchor
  converts 6/6 **once**; best-fit reaches 6/6 at **all fourteen**.
- **Enforced by** — ADR 0030 decision 5.
- **Cost** — "the anchor rescues the response" was a qwen3.8 finding stated as
  general.

### 2026-08-19 — `/api/generate` drops reasoning for gemma4 — and only gemma4
- **Evidence** — gemma4:31b-it-q4_K_M think-on: `/api/generate` returns **0**
  chars of reasoning, `/api/chat` returns **2021**, with `eval_count` **1281 vs
  1277** — generated either way, returned by one. nemotron3 returns **3909**
  chars on `generate`; qwen3.8 returns reasoning there normally.
  Token counts are **identical** across endpoints (1511/1511 think-off,
  1514/1514 think-on): ollama templates on `generate` too.
- **Enforced by** — `endpoint_compare.py`, suite default flipped to chat.
- **Cost** — I first wrote the rationale as "chat adds template tokens", which is
  false, and then as "`/api/generate` is behind", which over-generalised from one
  model. Both were reasoning, not measurement.

### 2026-08-19 — think-on is safe for the PINNED bbox task, and my blanket "think off" was over-broad
The think-on failures in this corpus are concentrated in multi-image
cross-referencing and under a `real` pin — not in the pinned, single-image,
anchored bbox request.

- **Evidence** — under the §0 configuration: qwen3.8 **14/14** geometries convert
  6/6 in *both* think modes; qwen3.6 **14/14** off and **13/14** on, its single
  failure being `sq320` (320×320), the geometry where both C7 checks lose
  discriminating power. Against that: qwen3.6 think-on on `multi_3img` does not
  terminate at all, and under a `real` pin its anchor converts at 1 of 14.
- **Enforced by** — SPEC §0.1, which states the permission and the four
  conditions it does *not* extend to.
- **Cost** — I recommended "think OFF" as a flat rule from the multi-image
  evidence. That would have given up thinking on a task where it is free.

### 2026-08-19 — Named coordinates fix the gemma4 axis flip, measured on the family that flips
- **Evidence** — `gemma4:26b-a4b-it-q4_K_M`, the MoE variant from the family
  that produced **all 11** measured `yxyx`-while-declaring-`xyxy` flips, emitted
  `norm1000`/`xyxy` and honoured it in **14/14** geometries in **both** think
  modes. `gemma4:31b-it-q4_K_M` likewise 14/14 × 2. The pin now stands at
  **111 of 112** cells across four models.
- **Enforced by** — SPEC C2 and §0.1.
- **Cost** — none; this confirms a clause rather than correcting one. Worth
  logging because it is the first time C2 was tested on the family it was written
  for, and because it does NOT clear `box_2d` — the positional form was never
  requested, so the flip risk is avoided rather than measured absent.

### 2026-08-19 — `box_2d` is honoured by both gemma4 models when the order is requested
- **Evidence** — the positional twin of the named-coordinate arm, one variable
  changed, 14 geometries × 2 models × 2 think modes: **56/56** cells used
  `box_2d`, declared `xyxy`, and `hits_declared` was **6/6 in every cell** —
  the declaration was TRUE, matching `hits_anchor` and `hits_bestfit`. Zero
  flips, including on the 26b MoE, the family that produced all 11 prior flips.
  IoU 0.841–0.978 (26b), 0.781–0.964 (31b).
- **Enforced by** — `bbox_contract_box2d_1img`.
- **Caveat that must travel with the number** — the prompt states
  `"coord_order": "xyxy"` in the requested shape, so this measures *"does gemma4
  honour an explicitly requested order on `box_2d`"*, NOT *"what order does it
  volunteer"*. The original 11 flips came from differently-worded arms. Whether
  the fix is the explicit order, the per-object declaration, the anchor or the
  build is **not** separated by this run.
- **Cost** — none; this closes a gap the named-coordinate sweep could not, since
  that sweep avoided the positional form rather than testing it.

### 2026-08-19 — REFUTED: the norm-1000 pin does not fix `scene_single` runaway. It causes it.
Stated as confirmed, retracted the same day, then refuted outright by the
corrected experiment. Recorded in full because the retraction alone would hide
that the claim had the SIGN backwards.

- **What was claimed** — `scene_single` asks for ABSOLUTE PIXEL coordinates, which
  has no closing condition; pinning norm-1000 recovered gemma4:26b-a4b think-on
  from IoU 0.334 to 0.972 with 4× less reasoning.
- **Why that was wrong** — the `pinned` arm changed TWO variables. It swapped the
  convention *and* dropped "The image is exactly {w} pixels wide and {h} pixels
  tall". The recovery came from removing the dimensions, not from the pin.
- **The corrected single-variable experiment** (dimensions retained in all three
  arms, verified programmatically before launch), 4 models × 2 think modes:

  | model | arch | mode | baseline (px) | pinned | anchored |
  |---|---|---|---|---|---|
  | gemma4:26b-a4b | **MoE** | on | 0.334 | **0.000** | 0.711 |
  | qwen3.6:35b-a3b | **MoE** | on | **0.971** | **0.044** | 0.640 |
  | nemotron3:33b | **hybrid** | on | 0.753 | 0.434 | 0.499 |
  | gemma4:31b | dense | on | 0.962 | 0.964 | 0.749 |
  | qwen3.8:27b | dense | on | 0.973 | 0.981 | 0.962 |

  The pin is **worse than baseline in 3 of 5 models**, catastrophically so in two.

  **The split tracks ARCHITECTURE, not baseline quality — pattern at n=5, not a
  mechanism.** Both MoE models and the hybrid are damaged; both dense models are
  not. Baseline quality does not predict it: qwen3.6 starts at a healthy 0.971
  and the pin takes it to 0.044, while gemma4:31b starts at 0.962 and is
  untouched. The gemma4 pair is the cleanest evidence — same family, same
  quantisation, same host, same prompt, differing only MoE vs dense: **0.334 vs
  0.962** on the baseline, and the pin drives the MoE to 0.000 while leaving the
  dense one at 0.964.

  Recorded as an OBSERVATION, explicitly not a conclusion: five models, one task,
  one scorer, and no account of why sparse routing would interact with a
  normalized coordinate space under thinking. Given four over-generalisations
  logged today, this is the shape of claim that needs another arch pair before it
  is believed.
  gemma4 pinned burned the full **122,880**-token budget for IoU 0.000. Think-off
  is unaffected everywhere (0.973 / 0.975 / 0.870 / 0.977), so this is a
  reasoning-time effect.
- **What actually survives** — (a) gemma4:26b-a4b think-on on scene is genuinely
  bad, 0.334 against its own 0.973 think-off, and NO prompt variant fixes it;
  anchored reaches only 0.711. That is a model property. (b) `qwen3.6` has **no**
  scene problem — baseline think-on is 0.971. Its runaway is specific to
  `multi_3img`, and "qwen3.6 runs away under thinking" was over-broad.
  (c) Stating a pixel size AND a normalized space appears worse than either
  alone: two frames to reconcile, and under thinking it deliberates until the
  budget dies.
- **Enforced by** — `scene_single_pinned` / `scene_single_anchored` retained as
  the evidence; full tables and the process post-mortem in
  [vision-campaign-2026-08-20-scene-termination.md](vision-campaign-2026-08-20-scene-termination.md).
  `scene_single` unchanged.
- **Cost** — a wrong claim was committed, pushed, and readable for ~2 hours; the
  correction needed two full 4-model runs. The cheap guard that would have caught
  it is the one now used: assert the single-variable property with code before
  launching, not in prose afterwards.

### 2026-08-19 — Persisting the reasoning text is what made the above diagnosable
- **Evidence** — the repetition counts, the identical re-derivations and the
  coordinate drift are all in `think_<tag>_<probe>.txt`, a file that did not
  exist this morning. Before it, a runaway cell was indistinguishable from an
  empty one: `eval_count` high, answer absent, cause unknowable.
- **Enforced by** — `client.persist()`, SPEC H9.
- **Cost** — every think-on cell measured before today discarded its reasoning,
  so none of them can be diagnosed retrospectively.

---

## 2026-08-19 — method

### 2026-08-19 — A capped cell is a floor, not a cost
`eval_count == num_predict` establishes "needs more than X", never "costs X".

- **Evidence** — at `num_ctx` 65536 the cell reported 57,344; at 131072 it
  reported 122,880. Doubling the ceiling changed the number and nothing else.
- **Enforced by** — [ADR 0012](adr/0012-benchmark-report-templates.md) rule 8;
  `CTX_LADDER` now ends at 131072.
- **Cost** — a diagnostic was one restatement away from being quoted as a cost.

### 2026-08-19 — Valid JSON is not usable JSON
- **Evidence** — the qwen3.6 q8_0 cell **parses cleanly** under `json.loads`;
  the model had serialised `answers` into a *string inside an unrelated array*.
  Scored 0/3 with chart 5/5. Recovered values (`q1: 2`, `q2: Q4/128`) match the
  same model's passing arm **exactly**. Rescoring 18 cells flips **exactly 3**;
  every other cell is byte-stable.
- **Enforced by** — `salvage.py`, SPEC C12 amendment, `salvage_method` per cell.
- **Cost** — a model failure was published that never happened, and the
  conclusion drawn from it ("the anchor makes q8_0 worse") was backwards.

### 2026-08-19 — 1920×1080 was never a clean control
- **Evidence** — `1080/32 = 33.75`, `1080/48 = 22.5`. Misaligned for **every**
  arch in the corpus; the fixture carried a 0.7%–2.2% pad in every measurement
  ever taken.
- **Enforced by** — SPEC C15 (alignment twins derived from `patch_stride`).

### 2026-08-19 — Contention corrupts measurements in both directions
- **Evidence** — loadavg **148** while a peer session ran timing-calibrated
  tests. Their measurements were invalidated; so were my `gen_tps`/`prefill_tps`
  fields, which I was about to publish as comparable.
- **Enforced by** — `endpoint_exclusive` in preflight; throughput omitted from
  the geometry campaign with the reason stated.

---

## 2026-08-19 — harness engineering

### 2026-08-19 — Duplication concentrates in the hardest part
- **Evidence** — **5** files had grown their own request code and had already
  diverged on 4 axes: endpoint default, `thinking` normalisation, response and
  reasoning persistence, context-overflow 400 translation. `finetext_probe` had
  **all four**, and its scores looked ordinary throughout.
- **Enforced by** — SPEC H9, `client.py`.

### 2026-08-19 — Consolidation must not normalise a calibrated payload
- **Evidence** — collapsing `send_think` into a boolean made
  `client.generate()` unable to emit `"think": true`, silently turning
  `variants.py`'s think-on arm into "whatever the server defaults to".
  `measure.py`'s payload had to stay byte-identical; its **19/19** text prefix is
  the proof it did.
- **Enforced by** — tri-state `send_think`, `num_ctx=False`, `use_env_opts`,
  `apply_sampling`; `test_client.py`.

### 2026-08-19 — Score-level tests give zero signal on the wire
- **Evidence** — **2 blockers** shipped green through **57** existing tests. One
  (`finetext_probe` `NameError`) fired *after* the inference completed, so the
  model was paid for and the result discarded; under `set -eu` it aborts the
  campaign.
- **Enforced by** — `test_client.py`, **25** tests asserting payload shape.
  Verified by reverting the fix and watching the test fail.
- **Cost** — an import-only check reported "imports OK" on a function that could
  not complete a single call.

### 2026-08-19 — Server-side knobs do not exist on hosts you do not own
- **Evidence** — `OLLAMA_MAX_LOADED_MODELS=1` is mandated by SPEC
  apple-silicon-build and is fixed when the *server* starts. 10.8.0.6 serves
  **5** ports. This host was previously driven to **106 GB** used and **53.89 GB**
  swap by a model left resident.
- **Enforced by** — [ADR 0031](adr/0031-model-residency-is-managed-client-side-on-remote-hosts.md),
  SPEC H10, `client.evict_others()` / `evict_all()`.
- **Also** — `keep_alive: 0` returns on *acceptance*, not completion, so the
  wait is load-bearing: without it a sweep holds two models at once.

### 2026-08-19 — Retry transport, never a deterministic rejection
- **Evidence** — a container restart killed rep 3 after **~23 minutes** of
  generation, when reps 1–2 were already byte-identical. A context-overflow 400
  fails identically forever; retrying it costs **50 s** and changes nothing.
- **Enforced by** — ADR 0031 decision 6; backoff 5/15/30 s; test asserts
  `urlopen` is called **exactly once** on a 400.

### 2026-08-19 — The context ladder multiplies cost across ALL arms, not just the capped one
- **Evidence** — one non-terminating arm (`scene_single`) forced gemma4:26b-a4b
  think-on from 16384 → 32768 → 65536, and each rung re-runs the **full 16-arm
  suite**. Think-off completed all 16 arms in **2.4 min**; the same model
  think-on spent 8.5 min at 16384, 10.3 min at 32768, and 35+ min at 65536.
- **Enforced by** — nothing yet. Recorded because it changes how a sweep should
  be scoped: isolating a known non-terminating arm into its own run is far
  cheaper than letting it drag every other arm up the ladder with it.

---

## 2026-08-19 — recurring failure patterns (mine)

Grouped because they share one cause: **acting on a mental model instead of
checking actual state.**

| pattern | occurrences | evidence |
|---|---|---|
| Watched a marker file instead of the real completion condition | **3** | watchers left polling sentinels with no writer — 2h51m, 3h20m, and one on a log the killed run never wrote |
| Published a number before the run that produced it landed | **2** | "53 of 54" (true: 55 of 56); frame ranges quoted as the sub-1.0 subset |
| Stated a rationale I had not measured | **2** | "chat adds template tokens" (identical); "`/api/generate` is behind" (gemma4 only) |
| Generalised from one model | **4** | anchor rescues; endpoint drops reasoning; "suspect the convention" published from n=1 and falsified at n=4 within the hour |
| Edited a module while a sweep ran against it | **1** | `NameError` mid-import, 1 cell lost |
| Stated an intended action as a completed one | **2** | said "restarting the full suite now", and later "applying the patch and running now", without issuing either command. Second one idled the GPU ~2h. Guard: launch and `pgrep`-verify in the SAME command |
| Changed two variables while calling it a single-variable arm | **1** | `scene_single_pinned` dropped the image-dimension sentence along with the convention; produced a 0.088 that looked like a model failure |
| Over-broad regex deletion | **1** | 4 prompt constants removed with the function being replaced |

**What actually caught these:** the adversarial verification pass, not the test
suite and not review-by-reading. It found 2 blockers and 3 wrong published
numbers that 57 tests and my own re-reading had missed.

**The cheapest guard found so far** is re-deriving a number from disk immediately
before writing it down. It caught the frame-range error in this very document.

---

## 2026-08-28 — model behaviour

### 2026-08-28 — gemma4:12b-nvfp4 think-on runs away on bbox arms under ON-policy sampling
On-policy sampling does not terminate this model's bbox reasoning on the
0.33.0 build. The 2026-08-13 runaway finding blamed off-policy sampling and
was measured on `0.32.5-maxusai-31a7f1ef`/b10353 for other models
([runaway-reasoning-under-think.md](runaway-reasoning-under-think.md)); this
observation is the sanctioned config (`card:gemma4+temp0`: temperature 0,
top_p 0.95, top_k 64) failing to save gemma4:12b-nvfp4 on
`0.33.0-maxusai-21cfe88e` (b10488, MLX 27fec909, Metal). Scoped to this
build + model + arms; not a refutation of the sampling finding.

- **Evidence** — campaign `mlx0330nv`, cell `mlx0330nv1_gemma4_12b-nvfp4_thinkon`:
  25 of 27 arms capped at 8,192 (rung 16384); after climbing to 65536
  (num_predict 57,344), **20 of 27 still capped** — every `bboxm_*` and
  nearly every `bbox_contract_*` arm. Only 7 arms ever finished:
  `scene_single_pinned`, `document_single`, `multi_3img`,
  `multi_3img_anchored`, `bboxm_pin_noanc_named`, `bbox_contract_anchored`,
  `finetext`. The 131072 ceiling attempt was abandoned mid-rung by operator
  descope on 2026-08-28 — the cell had consumed ~9.5 h wall without
  converging. Scores file kept as-is: 7 finished rows plus 20 capped rows
  whose highest recorded rung is 65536 (ADR 0012 conv 9 — capped rows are
  unfinished measurements; the file must not be summarized as results).
- **Enforced by** — `DESCOPED_CELLS` in `summarize_engine_compare.py`, which
  declares `(gemma4:12b-nvfp4, "on")` not worth measuring (2026-08-30). The
  driver skips the cell at every rung and a render prints `⚠ DESCOPED` rather
  than counting its absent arms as incomplete, so the ~9.5 h is never spent
  again by accident. This entry is the argument the declaration points back to.
  Scoped to that TAG: other gemma4:12b quantisations are not covered. Reopening
  it means deleting the entry and budgeting a full day at the 131072 ceiling.
- **Cost** — ~9.5 h wall clock (22:08 → 07:40) and four cold ladder rungs
  for one cell that produced 7 scores; the other nine campaign cells waited
  behind it all night.

### 2026-08-30 — The anchor fails to bound qwen3.6 runaway on Metal nvfp4 (n=1)
[2026-08-19](#2026-08-19--qwen36-think-on-does-not-terminate-the-anchor-bounds-it)
measured `multi_3img` failing to terminate while `multi_3img_anchored`
finished in 10,910 tokens, on two platforms and two builds — **for the
q4_K_M quant**. That campaign's nvfp4 row was Apple-only and already
imperfect (`✅✅❌` across three reps — the `q4_bbox` miss). On
`0.33.0-maxusai-21cfe88e` (Metal, nvfp4) **both members of the pair fail to
terminate** — one unrepeated observation. Read together: the mitigation was
imperfect on this quant at n=3 and is absent at n=1 here, so it must not be
relied on as a property of the model; whether it fails *outright* on nvfp4
needs repeats this campaign did not buy (its own limits section says the
same).

- **Evidence** — campaign `mlx0330nv`, cell
  `mlx0330nv1_qwen3_6_35b-a3b-nvfp4_thinkon`: 9 of 27 arms capped at 65536 and
  again at the 131072 ceiling, each producing **122,880 tokens** and **0
  answer characters** at ~70 tok/s. The nine include both halves of two pairs
  whose anchored member previously converged — `multi_3img` +
  `multi_3img_anchored`, `scene_single` + `scene_single_anchored` — plus
  `bboxm_pin_anc_pos`, `bboxm_free_anc_named`, `bboxm_free_noanc_named`,
  `bbox_contract_reasoning`, `bbox_contract_real_1img`. The remaining 18 arms
  converged — 9 at the 16384 start rung, 1 at 32768, 8 at 65536, the split
  the capped counts below imply (the scores file's per-arm `req_num_ctx` is
  the check) — max `eval_count` 22,823. So this is arm-specific
  non-termination, not a dead cell.
- **The ceiling rung resolved exactly zero arms** — capped counts by rung were
  18 → 17 → **9 → 9**. Escalation genuinely works up to 65536 (nine arms
  converted from capped to scored), and then stops working completely: the
  65536 → 131072 doubling bought 4.5 h of inference and not one measurement.
  That flat step is the cleanest signal available that an arm is
  non-terminating rather than under-budgeted, and it is worth reading off the
  `##### CAPPED` lines *during* a run instead of after.
- **Not a refutation of the anchor's value generally** — five `bboxm_*` arms
  including anchored ones did converge here, and 2026-08-19's measurement
  stands for its own build/quant. What is refuted is treating "add the anchor"
  as a *reliable* runaway bound across builds.
- **Enforced by** — `ladder_not_converged_at: 131072` stamped into all nine
  blocks; `ceiling-standing` now exits 0 for this cell, so future campaigns
  skip it outright unless `CTX_MAX` is raised.
- **Cost** — ~8 h wall (16:29 → 00:34) for nine unscored arms. Because
  `scene_single*` and `multi_3img*` are among them, this build has **no**
  think-on scene-grounding or multi-image reading for qwen3.6 at all.

### 2026-08-30 — The ladder earns its cost, and the arm-level resume is why
`gemma4:26b-nvfp4` think-on is the case that justifies the whole escalation
mechanism: **8 arms capped at the 16384 start rung, and 7 of them converged on
the way up** — capped counts by rung 8 → 4 → 2 → **1**. Without the ladder the
cell would have reported 8 unscored arms; with it, 26/27 are real
measurements (max `eval_count` 6,869) and exactly one arm,
`multi_3img_anchored`, is a genuine non-terminator standing at the ceiling
with 122,880 tokens and 0 answer characters.

- **The cost is bounded by the arm-level resume, not by the cell.** Each rung
  re-ran only the still-capped arms and printed what it skipped — `SKIP 19` /
  `SKIP 23` / `SKIP 25` already-scored arms against `RE-RUN 8` / `4` / `2`. So
  the climb re-ran 8+4+2 = 14 arms on top of the 27-arm first rung — 41
  arm-runs against a naive sweep's 4×27 = 108. This is the
  concrete answer to
  [2026-08-19's "the ladder multiplies cost across ALL arms"](#2026-08-19--the-context-ladder-multiplies-cost-across-all-arms-not-just-the-capped-one):
  it no longer does, and the `SKIP`/`RE-RUN` lines are the evidence to check
  when a climb looks too expensive.
- **The anchor's ordering flips here — one arm each way.** For 26b the
  *unanchored* `multi_3img` converged at the ceiling while
  `multi_3img_anchored` did not — the opposite ordering to
  [2026-08-19's qwen3.6 pair](#2026-08-19--qwen36-think-on-does-not-terminate-the-anchor-bounds-it).
  The prior [per-model finding](#2026-08-19--the-anchors-value-is-per-model-and-inverts)
  was powered at n=14 geometries; this is one observation per member, which
  is the shape of noise as readily as of inversion — an observation to
  repeat, not confirmation. What needs no repeat: three images plus an
  anchored coordinate contract is the most
  reasoning-expensive shape the suite asks for, and it is the one arm that
  fails on both non-converging gemma4/qwen cells.
- **Worth testing** — whether `ONLY_TESTS` isolation of a known-runaway arm is
  cheaper than letting it pull a cell up the ladder; at 26b's numbers the
  climb cost ~4.5 h to convert 7 arms, which was worth it.
