# TASK: re-run the five-model CUDA baseline — full ladder, one campaign tag

**Opened:** 2026-08-20. **Status:** DONE 2026-08-20 12:22 — delta run
completed, all acceptance criteria met; see Outcome at the bottom. Root cause
found and fixed the same day (SPEC H4b); the re-run was scoped to the DELTA
below — determinism at temperature 0 (ADR 0012 conv. 4) makes re-measuring
finished cells pure duplication.

## Root cause — why the ladder never climbed

The idempotent-resume commit (`ae0631de`, 2026-08-20 00:38) taught
`vision_suite.py` to skip any arm already present without an `error` key — so
a **capped** arm counted as done. `run_engine_compare.sh` escalates by
re-invoking the suite at the next rung against the same tag; resume skipped
every arm, and the following capped-check passed because the stale 8192
eval_counts sat below the new rung's larger cap. The ladder no-oped silently.
Timeline proves it: `g4full1` (Aug 19, pre-resume) climbed to 131072;
`cudafull1` (Aug 20 00:24–01:30, post-resume) froze every capped think-on
cell at (16384, 8192). Fixed: resume now uses `arm_done` (present, no error,
**not capped** — ADR 0012 conv. 9's "a capped cell is an unfinished
measurement" applied to the harness itself), regression-tested by
`test_summarizers.py::TestResumeNeverSkipsCapped`, codified as SPEC H4b.

The five-model CUDA head-to-head (T2) is currently assembled from two
campaigns (`cudafull1`, `g4full1`) and contains think-on cells that were
never given the window ladder. Two rendering defects masked this until
2026-08-20 — capped cells printed as scores, and pre-H11 columns vanished
from the provenance footer. Both are fixed and tested
(`test_summarizers.py::TestT2CappedCells`, `::TestProvenanceFooter`), so the
gap now shows on every render: `capped` cells and a ⚠ MIXED footer. The
render is honest; **the measurement is still missing.**

## What is owed (the ladder, not the cap count)

A `capped` cell is an unfinished run, not a result (ADR 0012 conv. 9). The
owed number per cell is the score at the final CONTEXT-ladder rung the model
needed to terminate (ADR 0012 rule 8) — up to CTX_LADDER's 131072 ceiling,
with `capped` at the ceiling reported as documented non-termination.

Cells known to need the ladder (all think-on, all stopped at
`num_predict` 8192 / `num_ctx` 16384 with no escalation):

| model | cells capped at (16384, 8192) |
|---|---|
| qwen3.6:35b-a3b-q4_K_M | scene_single, multi_3img, multi_3img_anchored |
| nemotron3:33b-q4_K_M | finetext, multi_3img, multi_3img_anchored |

These published as `0.000` IoU, `❌ ❌ ❌ 0/5` and `0/0/0/0/0` before the fix.
qwen3.6 may not terminate at any rung (ADR 0012 rule 6 records this for two
probes) — if so, that is the result, stated per rung, not left as a bare cap.

## What else the re-run fixes

- **gemma4:26b-a4b-it-q4_K_M / gemma4:31b-it-q4_K_M** exist only as `g4full1`
  runs: pre-H11 (no `host`/`server_version` on any cell) — the columns the
  MIXED footer refuses to vouch for. The 26b think-on scene cell *did*
  terminate at the 131072 rung (eval 7576, IoU 0.334); that is a legitimate
  rule-8 result, but it needs re-measuring under the campaign tag with
  provenance so it can sit in the same table.
- One campaign tag for all five models ends the `cudafull1`+`g4full1` split
  that made the table MIXED in the first place.

## Run — the delta only

Everything already finished stays: same host, same build
(`0.32.14-rc0-dynres-0-ga5d6590`, re-verified 2026-08-20 before launch), and
temperature 0 makes a re-measure byte-identical. The fixed resume (H4b) is the
mechanism: re-running a tag re-runs exactly the capped arms. Tags stay
`cudafull1_*` — `TAG_PREFIX="cudafull"` plus the interpolated `rep=1`
reproduces them, so the campaign stays ONE tag namespace and the gemma pair
joins it (their new files carry H11 provenance; the pre-H11 `g4full1` files
remain as history). Stash the current score files first (standing convention).

The scoped think-on runs start at `NUM_CTX=32768`: the 16384 rung is already
measured for those cells — capped, deterministically — so starting above it
skips a known duplicate. H4a holds (65536 and 131072 remain above the start).

```bash
cd docs/maxusai/vision-suite
# 1) nemotron think-on: the three capped T2 cells, ladder from 32768
MODELS="nemotron3:33b-q4_K_M" THINK_MODES="on" TAG_PREFIX="cudafull" NUM_CTX=32768 \
  ONLY_TESTS="finetext,multi_3img,multi_3img_anchored" \
  ./run_engine_compare.sh http://10.8.0.6:11497
# 2) gemma pair: full suite, both modes, default ladder (no valid cells exist)
MODELS="gemma4:26b-a4b-it-q4_K_M gemma4:31b-it-q4_K_M" TAG_PREFIX="cudafull" \
  ./run_engine_compare.sh http://10.8.0.6:11497
# 3) qwen3.6 think-on: the three capped T2 cells, ladder from 32768 (may not
#    converge below the 131072 ceiling — that is then the documented result)
MODELS="qwen3.6:35b-a3b-q4_K_M" THINK_MODES="on" TAG_PREFIX="cudafull" NUM_CTX=32768 \
  ONLY_TESTS="scene_single,multi_3img,multi_3img_anchored" \
  ./run_engine_compare.sh http://10.8.0.6:11497
```

Known residuals deliberately NOT in scope: capped diagnostic arms that no
template renders (nemotron `scene_single_anchored`, `bbox_contract_adv_real`;
19 further `bboxm_*`/`bbox_contract_*` arms on qwen3.6 think-on). They stay
capped in the score files — the fixed resume will pick them up if a later
investigation needs them laddered; do not let them ride along here, a
non-terminating qwen3.6 arm costs ~205k tokens to prove non-convergence.

**Residuals laddered after all** (maintainer directive "run uncapped",
2026-08-20 afternoon; `run_cudafull1_residual_2026-08-20.log`, stash
`stash/2026-08-20-pre-residual-ladder/`). Nemotron's two arms converged at
32768. qwen3.6's 19 arms: 13 converged at 32768, 4 at 65536
(`bboxm_pin_noanc_pos`, `scene_single_pinned`, `scene_single_anchored`,
`bbox_contract_adv_real`), and **2 did not converge at the 131072 ceiling** —
`bbox_contract` and `bbox_contract_real_1img`, each burning the full 122,880
budget, recorded with the server's own `done_reason: length` (the first
production verdicts after PR #207). Final campaign state: across all ten
`cudafull1` files the only capped cells are qwen3.6 think-on `multi_3img`,
`bbox_contract`, `bbox_contract_real_1img`, all AT the ceiling — three
documented non-terminating probes, zero unfinished measurements below it.

## Acceptance

- T2 renders both think modes with **zero `capped` quality cells**, except
  cells capped at the 131072 ceiling, which the report names as
  non-termination per model and probe.
- Footer: one host, one build, **no ⚠ MIXED**, no pre-H11 entries.
- Every quality cell carries its final-rung `(num_ctx)`; rungs may differ
  between cells — that is rule 8 working, not a defect.
- Tables pasted verbatim from the generators (H7), both think modes.

## Outcome — 2026-08-20, delta run 10:29–12:22 AEST (1h53m)

Every T2 quality cell now carries a final-rung result; the one remaining
`capped` cell is qwen3.6 think-on `multi_3img` at the 131072 CEILING —
eval_count 122880 == num_predict, i.e. **documented non-termination**
(ADR 0012 conv 9): the model will not stop thinking on this probe at any
rung in the ladder. Rendered by the generator, pasted verbatim (H7):

### T2 think=false

| test | metric | cudafull1_gemma4_26b-a4b-it-q4_K_M_thinkfalse | cudafull1_gemma4_31b-it-q4_K_M_thinkfalse | cudafull1_qwen3_8_27b-q4_K_M_thinkfalse | cudafull1_nemotron3_33b-q4_K_M_thinkfalse | cudafull1_qwen3_6_35b-a3b-q4_K_M_thinkfalse |
|---|---|---|---|---|---|---|
| scene | bbox IoU | 0.973 (16384) | 0.962 (16384) | 0.977 (16384) | 0.870 (16384) | 0.975 (16384) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.756 (16384) | 0.752 (16384) | 0.550 (16384) | 0.044 (16384) | 0.607 (16384) |
| fine text | 22/16/12/9/7 px | 4/4/4/3/3 (16384) | 4/4/4/4/3 (16384) | 4/4/4/2/1 (16384) | 4/4/4/3/0 (16384) | 4/4/4/2/2 (16384) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ❌ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) |
| throughput | gen tok/s | 150 | 56 | 65 | 209 | 95 |
| throughput | prefill tok/s | 1202 | 646 | 1799 | 4797 | 3228 |
| latency | s/req (unique image) | 5.0 | 12.2 | 9.8 | 3.0 | 6.6 |
| latency | req/h (serial) | 725 | 296 | 366 | 1196 | 545 |

Provenance (from score files): host(s) http://10.8.0.6:11497 · build(s) 0.32.14-rc0-dynres-0-ga5d6590

### T2 think=on

| test | metric | cudafull1_gemma4_26b-a4b-it-q4_K_M_thinkon | cudafull1_gemma4_31b-it-q4_K_M_thinkon | cudafull1_qwen3_8_27b-q4_K_M_thinkon | cudafull1_nemotron3_33b-q4_K_M_thinkon | cudafull1_qwen3_6_35b-a3b-q4_K_M_thinkon |
|---|---|---|---|---|---|---|
| scene | bbox IoU | 0.334 (65536) | 0.962 (16384) | 0.975 (16384) | 0.577 (16384) | 0.717 (32768) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.714 (16384) | 0.709 (16384) | 0.858 (16384) | 0.114 (16384) | 0.401 (16384) |
| fine text | 22/16/12/9/7 px | 4/4/4/4/3 (16384) | 4/4/4/4/3 (16384) | 4/4/4/1/0 (16384) | 4/4/4/4/0 (65536) | 4/4/4/2/2 (16384) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ❌ 5/5 (16384) | ✅ ✅ ❌ 5/5 (32768) | capped (131072) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (32768) | ✅ ✅ ✅ 5/5 (32768) |
| throughput | gen tok/s | 140 | 56 | 66 | 230 | 88 |
| throughput | prefill tok/s | 417 | 156 | 1543 | 2251 | 1546 |
| latency | s/req (unique image) | 58.1 | 35.2 | 19.2 | 24.3 | 242.0 |
| latency | req/h (serial) | 62 | 102 | 188 | 148 | 15 |

Provenance (from score files): host(s) http://10.8.0.6:11497 · build(s) 0.32.14-rc0-dynres-0-ga5d6590

Exploratory render (rule 7) of the re-laddered arms, emitted by
`render_residuals.py` from the score files:

### nemotron3:33b-q4_K_M think=on

| arm | final rung (num_ctx) | eval tok | done_reason | result |
|---|---|---|---|---|
| scene_single_anchored | 32768 | 13628 | — (pre-#207 block) | IoU 0.417, labels 6/6, serial ❌ |
| bbox_contract_adv_real | 32768 | 9924 | — (pre-#207 block) | hits 0/6 declared, contract ❌, dialect norm1000/xyxy |

### qwen3.6:35b-a3b-q4_K_M think=on

| arm | final rung (num_ctx) | eval tok | done_reason | result |
|---|---|---|---|---|
| bboxm_pin_anc_named | 32768 | 14042 | — (pre-#207 block) | hits 6/6 declared, contract ✅, dialect norm1000/xyxy |
| bboxm_pin_anc_pos | 32768 | 15542 | — (pre-#207 block) | hits 4/6 declared, contract ❌, dialect norm1000/xyxy |
| bboxm_pin_noanc_named | 32768 | 21500 | — (pre-#207 block) | hits 6/6 declared, contract ✅, dialect norm1000/xyxy |
| bboxm_pin_noanc_pos | 65536 | 9453 | stop | hits 6/6 declared, contract ✅, dialect norm1000/xyxy |
| bboxm_free_anc_named | 32768 | 13461 | — (pre-#207 block) | hits 6/6 declared, contract ✅, dialect norm1000/xyxy |
| bboxm_free_anc_pos | 32768 | 15070 | — (pre-#207 block) | hits 3/6 declared, contract ❌, dialect norm1000/xyxy |
| bboxm_free_noanc_pos | 32768 | 18586 | — (pre-#207 block) | hits 6/6 declared, contract ✅, dialect norm1000/xyxy |
| scene_single_pinned | 65536 | 4792 | stop | IoU 0.044, labels 6/6, serial ✅ |
| scene_single_anchored | 65536 | 8423 | stop | IoU 0.640, labels 6/6, serial ✅ |
| bbox_contract | 131072 | 122880 | length | capped — non-terminating at the ceiling |
| bbox_contract_multi | 32768 | 17075 | — (pre-#207 block) | hits 6/6 declared, contract ✅, dialect real/xyxy |
| bbox_contract_reasoning | 32768 | 8290 | — (pre-#207 block) | hits 5/6 declared, contract ❌, dialect norm1000/xyxy |
| bbox_contract_perobject | 32768 | 16638 | — (pre-#207 block) | hits 6/6 declared, contract ✅, dialect norm1000/xyxy |
| bbox_contract_anchored | 32768 | 11902 | — (pre-#207 block) | hits 6/6 declared, contract ✅, dialect norm1000/xyxy |
| bbox_contract_anchored_1img | 32768 | 13932 | — (pre-#207 block) | hits 6/6 declared, contract ✅, dialect norm1000/xyxy |
| bbox_contract_box2d_1img | 32768 | 14639 | — (pre-#207 block) | hits 6/6 declared, contract ✅, dialect norm1000/xyxy |
| bbox_contract_positional_1img | 32768 | 21154 | — (pre-#207 block) | hits 6/6 declared, contract ✅, dialect norm1000/xyxy |
| bbox_contract_real_1img | 131072 | 122880 | length | capped — non-terminating at the ceiling |
| bbox_contract_adv_real | 65536 | 14297 | stop | hits 5/6 declared, contract ❌, dialect norm1000/xyxy |

Provenance (from score files): host(s) ['http://10.8.0.6:11497'] · build(s) ['0.32.14-rc0-dynres-0-ga5d6590'] · think=on, ladder 32768→131072, temp per sampling.py

The canonical T1 campaign matrix, renderable since
`summarize_engine_compare.py --prefix` landed (same day). Its quality
cells carry the conv-9 `capped` guard; the throughput table carries the
anchored-multi column beside the unanchored one (the pair distinguishes
a frame error from a grounding failure) and **Think tok** —
token_split.py's exact reasoning-token count, gate-proven against
eval_count with each server's own vocab (nemotron3 renders `—`: no
local tokenizer passed the gate; a GGUF-array reconstruction was
refused at 54/54 cells, which is the gate working):

### T1 think=false

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384 | 0.973 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 0.962 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 0.977 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 0.870 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 16384 | 0.975 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | 0 | 536 | 150 | 1202 | 5.0 | 725 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | 0 | 538 | 56 | 646 | 12.2 | 296 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 2 | 1 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | 0 | 544 | 65 | 1799 | 9.8 | 366 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 209 | 4797 | 3.0 | 1196 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 2 | 2 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | 0 | 548 | 95 | 3228 | 6.6 | 545 |

Provenance (from score files): host(s) http://10.8.0.6:11497 · build(s) 0.32.14-rc0-dynres-0-ga5d6590 · think=false

### T1 think=on

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384/65536 ⚠ | 0.334 | 0/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 0.962 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 0.975 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| nemotron3:33b-q4_K_M | GGUF | 16384/32768/65536 ⚠ | 0.577 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 16384/32768/131072 ⚠ | 0.717 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Gen tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384/65536 ⚠ | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | 7153 | 7576 | 140 | 417 | 58.1 | 62 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | 833 | 1372 | 56 | 156 | 35.2 | 102 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 1 | 0 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | 690 | 1148 | 66 | 1543 | 19.2 | 188 |
| nemotron3:33b-q4_K_M | GGUF | 16384/32768/65536 ⚠ | 4 | 4 | 4 | 4 | 0 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 5308 | 230 | 2251 | 24.3 | 148 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 16384/32768/131072 ⚠ | 4 | 4 | 4 | 2 | 2 | capped | ✅ q1 + q2 + q4-bbox | 20445 | 21058 | 88 | 1546 | 242.0 | 15 |

Provenance (from score files): host(s) http://10.8.0.6:11497 · build(s) 0.32.14-rc0-dynres-0-ga5d6590 · think=on

### Thinking tokens per probe (think=on) — exploratory (rule 7), from the H14 splits

qwen3.8's nine missing arms (its run predated the matrix commit that
added them to the default suite by six minutes) were measured
2026-08-20 evening: all nine converged at the 16384 start rung in both
think modes, splits gate-passed at [1, 4].

| probe (think=on) | gemma4:26b-a4b | gemma4:31b | qwen3.8:27b | qwen3.6:35b-a3b |
|---|---|---|---|---|
| bboxm_pin_anc_named | 1466 | 541 | 699 | 13531 |
| bboxm_pin_anc_pos | 3362 | 1690 | 762 | 14989 |
| bboxm_pin_noanc_named | 3406 | 337 | 1335 | 21058 |
| bboxm_pin_noanc_pos | 1889 | 1557 | 971 | 8975 |
| bboxm_free_anc_named | 2451 | 1258 | 431 | 12775 |
| bboxm_free_anc_pos | 3041 | 692 | 457 | 14343 |
| bboxm_free_noanc_named | 1508 | 729 | 559 | 5585 |
| bboxm_free_noanc_pos | 2787 | 1343 | 629 | 18040 |
| scene_single_pinned | 1759 | 1836 | 1421 | 4184 |
| scene_single_anchored | 3556 | 2413 | 1079 | 7729 |
| scene_single | 7153 | 833 | 690 | 20445 |
| document_single | 3403 | 1992 | 616 | 2224 |
| multi_3img | 3658 | 1275 | 799 | capped (131072) |
| multi_3img_anchored | 5583 | 2209 | 1119 | 9598 |
| bbox_contract | 2302 | 1024 | 1160 | capped (131072) |
| bbox_contract_multi | 1825 | 5393 | 802 | 16757 |
| bbox_contract_reasoning | 2850 | 1408 | 940 | 7843 |
| bbox_contract_pinned | 1504 | 405 | 906 | 2619 |
| bbox_contract_perobject | 1176 | 405 | 1013 | 16226 |
| bbox_contract_anchored | 1862 | 891 | 1027 | 11391 |
| bbox_contract_anchored_1img | 1810 | 859 | 676 | 13404 |
| bbox_contract_box2d_1img | 3378 | 1047 | 844 | 14086 |
| bbox_contract_positional_1img | 3195 | 981 | 1064 | 20599 |
| bbox_contract_real_1img | 2112 | 894 | 607 | capped (131072) |
| bbox_contract_adv_real | 2676 | 1289 | 528 | 13640 |
| bbox_contract_adv_norm1 | 2385 | 916 | 616 | 4312 |
| finetext | · | 1809 | · | 1438 |
| **total (uncapped, split cells)** | **72097** | **36026** | **21750** | **275791** |

Provenance (from score files): host(s) ['http://10.8.0.6:11497'] · build(s) ['0.32.14-rc0-dynres-0-ga5d6590'] · thinking_tokens per SPEC H14 (token_split.py, gates [1,6]/[1,6]/[1,4]/[0,29])

### Fine-text small tiers: qwen3.8's deficit is optical, not reasoning

Why qwen3.8 reads `4/4/4/1/0` where gemma4 reads `4/4/4/4/3`: a
character-level diff of the returned codes against `finetext_gt.json`
(2026-08-20). qwen3.8 returns all 20 codes with correct structure, tier and
slot; the small-tier misses are 2–3 character glyph confusions inside
otherwise-correct codes — M↔N, W↔K, D↔O, V↔X/Y, digits 1↔3↔5↔9:

- 9px (2/4): `RNK-0391-DW18` → `RMK-0391-OW18`; `FFW-8248-UJ83` → `FRM-8248-UJ83`
- 7px (0/4): `NTR-7871-KK15` → `MTR-7871-KN13`; `AYK-9301-CK10` → `AVW-9301-CK30`;
  `PDX-3473-YF99` → `PDV-3473-YF59`; `WKW-6311-UR66` → `MNW-6312-UR66`

gemma4:31b on the identical image: 9px 4/4 exact; 7px 3/4, its one miss the
same kind (`WKW…` → `WXM…`). The mode is irrelevant — qwen3.8 think-off shows
the same tier profile (4/4/4/2/1) — so this is a vision-encoder resolution
limit at 7–9px glyphs, not fabrication, refusal, or a thinking failure (its
thinking stream is a clean tier-by-tier transcription plan). Plausible but
unproven mechanism: qwen3.8 demonstrably works in an internally rescaled
frame (~1.22× measured on the scene image, the q4 investigation above), and
resampling is exactly what smears 7px strokes.

The scoring caveat recorded with it: the diffed qwen3.8 codes are the PROBE
generation's answer (the suite generation's text was the double-persist
casualty); the suite generation scored 9px 1/4 vs the probe's 2/4 — same
failure mode, different draw (ADR 0012 conv. 4).
