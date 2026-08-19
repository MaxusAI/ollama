# ADR 0028: the benchmark harness has one runner and one set of helpers; arms extend it, they do not fork it

- **Status:** accepted 2026-08-17, on fork `main`. Enforced by
  [SPEC: vision harness reuse](../spec/vision-harness-reuse.md) (H1–H9; H9 extends this ADR to the request path, added 2026-08-19) and
  stated as an invariant in `AGENTS.md`.
- **Date:** 2026-08-17
- **Deciders:** MaxusAI fork maintainers
- **Related:** [ADR 0012](0012-benchmark-report-templates.md) (report
  templates — the same "generator, not by hand" principle applied to output),
  [ADR 0022](0022-thinking-is-off-for-vision-work.md) (the harness traps a
  bespoke loop reintroduces), [ADR 0011](0011-preflight-expectations-are-versioned-code.md)

## Context

`run_engine_compare.sh` accumulated a great deal of hard-won behaviour: a
`num_ctx` ladder with **per-cell** escalation, `num_predict` derived from the
rung as `num_ctx - CTX_PROMPT_RESERVE`, a cold server per cell rather than per
model, a powermode stamp per cell, and a prompt reserve sized from the measured
worst case. Every one of those exists because its absence produced a wrong
result that was published and later retracted.

In one week, **six bespoke loops** were written against `vision_suite.py`
directly — `multi_repeats`, `placement_ab`, `anchored_c`, `adversarial`,
`lowtemp`, `lowtemp31b` — each re-implementing the model loop, the cold restart
and the tagging. **None implemented the ladder, and none derived `num_predict`
for think-on.** Every one therefore ran think-on at the think-off default of
2200, and [ADR 0022](0022-thinking-is-off-for-vision-work.md)'s trap #1 says
exactly what that produces: budget exhaustion inside an unclosed thinking block,
an empty response, and a cell that reads as a vision failure. One arm's entire
output had to be discarded and re-run.

The same failure recurred in reporting. `summarize_reps.py` was written from
scratch and re-implemented four helpers that already existed in
`summarize_engine_compare.py`: `engine_for`, `was_capped`, `ctx_for` and
`tag_for`. One re-implementation was already wrong — the local capped test used
`eval_count == num_predict` where `was_capped` uses `>=`, so it would have
counted an overrunning cell as a scored result.

And in presentation: generator output was transcribed by hand into a campaign
document, dropping the `num_ctx` column, after which a mid-ladder read published
`nemotron3:33b-bf16` think-on as scene IoU **0.000** when the settled value was
**0.872 at num_ctx 32768**.

Three incidents, one cause: **rewriting something the repository already had.**

## Decision

**There is one runner, one set of report helpers, and one set of generators.
Work that needs something they do not do extends them; it does not fork them.**

1. **No bespoke loop over models.** `run_engine_compare.sh` is the only entry
   point that iterates models and think modes. It carries `REPEATS`,
   `TAG_PREFIX` and `ONLY_TESTS` precisely so that a repeated, subsetted or
   sampling-overridden **arm** runs through it and inherits the ladder, the
   escalation, the cold restart and the provenance stamp.
2. **A missing knob is a patch to the runner, not a new script.** If an arm
   needs behaviour the runner lacks, add the knob. One review, and every future
   arm inherits it — the same argument ADR 0012 makes for generators.
3. **Shared helpers are imported.** Anything a summarizer needs that
   `summarize_engine_compare.py` already defines — engine detection, the capped
   test, `num_ctx` extraction, tag construction, score loading — is imported
   from it. A second definition is a second thing to get wrong, and the first
   time this was violated the copy was already wrong.
4. **Tables are emitted by a generator and pasted verbatim** (ADR 0012 rules 1
   and 8), including into chat replies and PR descriptions. "Reformatting for
   readability" is what dropped the `num_ctx` column.
5. **Read before writing.** Before adding a script or a function to
   `docs/maxusai/vision-suite/`, check whether the behaviour exists. The
   inventory is in `vision-suite/README.md` §Files.

## Consequences

- Arms are one command with environment knobs, not a file. The low-temperature
  arm — six models, three repeats, both think modes, a subset of tests and a
  sampling override — is a single `run_engine_compare.sh` invocation.
- A fix to escalation, restart or provenance lands once and applies everywhere,
  including to arms written afterwards.
- Scratch scripts under a session scratchpad are still fine for one-off
  inspection and analysis. The rule binds anything that **produces measurements
  or renders results**, because those get published.
- The cost is that the runner grows knobs. That is deliberate: a knob is
  reviewable and shared, a private loop is neither.

## Alternatives considered

- **Let arms be bespoke; they are exploratory.** Rejected on evidence: six of
  six were wrong in the same way, and their output was published before anyone
  noticed. Exploratory work is exactly where a silent harness bug survives,
  because there is no baseline to contradict it.
- **A library of shell functions each script sources.** Rejected: it keeps the
  proliferation and only shares the easy parts. The ladder is control flow, not
  a helper — it cannot be factored into a function a script may forget to call.
- **Code review as the control.** Rejected as the *primary* control: all six
  loops were plausible on their face, and the defect was an absence rather than
  a mistake. Absences are what a shared entry point removes by construction.
