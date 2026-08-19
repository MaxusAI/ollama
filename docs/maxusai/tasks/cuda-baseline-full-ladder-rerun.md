# TASK: re-run the five-model CUDA baseline — full ladder, one campaign tag

**Opened:** 2026-08-20. **Status:** owed — nothing below is publishable until done.

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

## Run

All five models, both think modes, full ladder, one tag, CUDA host
`http://10.8.0.6:11497`:

```bash
cd docs/maxusai/vision-suite
MODELS="gemma4:26b-a4b-it-q4_K_M gemma4:31b-it-q4_K_M qwen3.8:27b-q4_K_M nemotron3:33b-q4_K_M qwen3.6:35b-a3b-q4_K_M" \
TAG_PREFIX="cudafull2-" ./run_engine_compare.sh http://10.8.0.6:11497
```

(Host is the positional argument; CTX_MAX already defaults to the 131072
ceiling. Runner details are authoritative in `run_engine_compare.sh` — H4a
refuses a think-on run whose CTX_MAX leaves no rung to climb; do not set
ALLOW_NO_LADDER.)

## Acceptance

- T2 renders both think modes with **zero `capped` quality cells**, except
  cells capped at the 131072 ceiling, which the report names as
  non-termination per model and probe.
- Footer: one host, one build, **no ⚠ MIXED**, no pre-H11 entries.
- Every quality cell carries its final-rung `(num_ctx)`; rungs may differ
  between cells — that is rule 8 working, not a defect.
- Tables pasted verbatim from the generators (H7), both think modes.
