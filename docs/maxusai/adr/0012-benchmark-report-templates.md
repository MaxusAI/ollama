# ADR 0012: benchmark reports use three canonical templates, rendered by generators

- **Status:** accepted, 2026-08-08 (maintainer directive "resolve the issues",
  same day as proposal). Validated in practice before acceptance: the
  2026-08-08 power campaign (PR #61/#62) was rendered exclusively through T1/T2
  with the latency pair and powermode provenance. Existing docs stay
  grandfathered; new reports use the templates.
- **Related:** [ADR 0011](0011-preflight-expectations-are-versioned-code.md)
  (same philosophy: facts live in versioned code, prose links to them),
  [vision-benchmark-baseline.md](../vision-benchmark-baseline.md) (T3 instance
  and the single source of metric definitions).

## Context

A survey of `docs/maxusai/` found **12+ distinct table shapes** for reporting
the same kinds of measurements (model comparisons, budget sweeps, throughput).
Two consequences, both observed this week: hand-built tables acquire
transcription errors (a serial ✓ that was ✗; image tokens derived with the
wrong text baseline), and readers cannot diff runs because each report invents
its own columns. The engine-compare campaign already solved this locally —
`summarize_engine_compare.py` renders tables from `scores_*.json` so a report
can never disagree with its data. This ADR generalizes that.

## Decision

Three templates cover every recurring report. **Any table whose numbers come
from harness output MUST be emitted by the named generator**, not typed.

| id | shape | when | generator |
|---|---|---|---|
| **T1 — Campaign matrix** | two tables: grounding+extraction, then OCR tiers + multi + throughput + latency; one row per model, `Engine` column, MLX bolded | full N-model campaigns on one host/power state | `vision-suite/summarize_engine_compare.py` (paired with `run_engine_compare.sh`) |
| **T2 — Head-to-head pivot** | rows = test × metric, columns = models | deep comparison of ≤ 4 configurations | `vision-suite/summarize_head_to_head.py` (same scores files; `--tags` for ad-hoc tag names) |
| **T3 — Platform baseline** | structured report: system-under-test, workloads, metric definitions, results, regression procedure, limitations | the living per-platform record | hand-maintained, but its table shapes are fixed and its §3 is the **only** place metric definitions live — other reports link, never restate |

Shared conventions, binding for all three:

1. **Provenance header** on every report: date, host, **power mode** (`pmset`
   powermode on macOS; `n/a` elsewhere), server version + payload + patchset,
   endpoint, sampling params. `run_engine_compare.sh` stamps powermode into the
   run log per model; the report header carries it per campaign.
2. **Latency pair is mandatory wherever tok/s appears**: `s/req` (unique-image
   steady state = scene decode + full prefill at clean rates) and `req/h`
   (3600 / s_req, serial). Rationale: tok/s alone flattered high-prefill
   configurations and hid the scaling story.
3. **Validity annotations, never silent cells**: `n/v` + footnote for
   overhead-dominated or cache-hit throughput; `—` for unmeasured; a struck or
   flagged cell for known-invalid (e.g. MLX vision-blind era). An empty cell is
   indistinguishable from "fine" and is forbidden.
4. **Determinism note**: quality cells are bit-reproducible at temperature 0
   per (payload, backend, budget, image); cross-image noise floor ±0.01 IoU.
   Reports state deviations, not re-derive them.
6. **Quality cells carry their `num_ctx` in brackets: `value (num_ctx)`.**
   `num_ctx` is per **model and per test**, not per campaign — measured maxima
   for a valid think-on run span 3,258 (gemma4 fine-text) to 16,421 (nemotron3
   document), and qwen3.6 does not terminate on two probes at any window tried.
   It belongs in the table because it changes what a result *means*: an empty
   response is "truncated by the window" at one value and "the model would not
   stop" at another, and a bare score cannot distinguish them. That ambiguity
   already produced a wrong published reading — cells recorded as reproducible
   model behaviour were context truncation, `eval_count` matching
   `num_ctx - prompt` exactly. `vision_suite.py` records `req_num_ctx` and
   `req_num_predict` per test so the generators render this without anyone
   re-deriving it; `(?)` marks runs predating those fields and is not a defect.

7. **Exploratory tables are exempt** from T1/T2 shapes (investigations need
   free-form arms) but still carry the provenance header and validity marks.

8. **When the window ladder escalates, the reported `num_ctx` is the final
   successful rung, and it is a first-class result — not a footnote.** A run that
   needed 65,536 to terminate has not performed the same as one that finished at
   16,384, however similar the scores look. Report the rung a model *reached*
   alongside its score; do not summarise escalation as prose ("the ladder
   climbed") and do not report cap counts in its place. **Cap counts are harness
   diagnostics, not results** — the reader wants the answer and what it cost to
   obtain, which is exactly `(score, num_ctx, num_predict)`.

   **The score file holds only the final rung.** Each escalation re-runs the
   whole suite and overwrites, so a file read mid-ladder shows a superseded
   result. Re-render after a campaign fully completes, including any resume, and
   never transcribe generator output by hand — measured 2026-08-17, a
   hand-copied table published `nemotron3:33b-bf16` think-on as scene IoU
   **0.000** (read at the 16,384 rung) when the settled value is **0.872** at
   32,768. Both the dropped column and the stale number came from typing what a
   generator had already rendered correctly, which is what rule 1 above exists
   to prevent.

## Alternatives considered

- **One universal table.** Rejected: the campaign matrix and the pivot answer
  different questions (breadth vs contrast); forcing one shape reproduces the
  ad-hoc drift this ADR ends.
- **Templates as documentation only (no generators).** Rejected: this week's
  transcription errors happened *with* the format already agreed; only
  generation from the score files prevents them (ADR 0011's argument).

## Consequences

- New metrics are added by extending a generator (one review, all future
  reports inherit it) — e.g. `s/req`/`req/h` landed as a generator change.
- Score files (`scores_*.json`, `ft_*.json`) become the report interface;
  their keys are as load-bearing as the tables and change with the same care
  as `expectations.toml`.
- Historical docs keep their shapes; re-rendering them is explicitly out of
  scope.
