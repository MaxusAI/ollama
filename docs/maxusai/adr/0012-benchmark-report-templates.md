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
5. **Exploratory tables are exempt** from T1/T2 shapes (investigations need
   free-form arms) but still carry the provenance header and validity marks.

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
