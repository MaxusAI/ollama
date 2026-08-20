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
