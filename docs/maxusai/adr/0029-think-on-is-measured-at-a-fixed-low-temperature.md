# ADR 0029: think-on is measured at a fixed low temperature, not at card sampling

- **Status:** accepted 2026-08-17. Supersedes the *sampling* half of
  [ADR 0023](0023-think-mode-is-per-model-and-measured-on-policy.md). ADR 0023's
  other rulings — that think mode is decided per model, and that a think-on
  regression claim is inadmissible without checking `eval_count` against
  `num_predict` — are carried forward unchanged.
- **Date:** 2026-08-17
- **Deciders:** MaxusAI fork maintainers
- **Related:**
  [vision-lowtemp-thinkon-negative-result.md](../vision-lowtemp-thinkon-negative-result.md)
  (the measured cost of this decision),
  [runaway-reasoning-under-think.md](../runaway-reasoning-under-think.md) (why
  card sampling was adopted in the first place),
  [ADR 0022](0022-thinking-is-off-for-vision-work.md) (the harness traps),
  [ADR 0012](0012-benchmark-report-templates.md) (±0.01 noise floor),
  [ADR 0011](0011-preflight-expectations-are-versioned-code.md)

## Context

ADR 0023 moved think-on off greedy decoding and onto card-sourced sampling,
because `temperature 0` made reasoning fail to terminate: every token landed in
`thinking`, `eval_count` hit `num_predict`, and `response` came back empty. For
gemma4 the card value is `temperature 1.0`. That fixed termination.

It also made think-on **irreproducible**. At `temperature 1.0` two runs of the
same model over the same fixture differ, and this suite can afford `n <= 3`. At
that sample size a difference between two engines, two quantisations or two code
revisions cannot be separated from sampling noise. The
[low-temperature arm](../vision-lowtemp-thinkon-negative-result.md) was run to
test a fix and recorded as a negative result; the arm that motivated it —
`contract_followed` at `n=3` — showed the problem directly, with per-repeat
values like `6/8/8` and `7/5/6` on a categorical metric.

The suite exists to gate regressions. A regression gate needs a number that is
the same when nothing changed. On-policy realism and reproducibility are in
direct conflict here, and only one of them is load-bearing for that job.

## Decision

**Think-on is measured at `THINK_TEMPERATURE`, default `0.01`, for every family.
Temperature is fork policy. Every other sampling field stays card-sourced.**

1. `sampling.py` supplies `temperature` from `THINK_TEMPERATURE`;
   `top_p`, `top_k`, `min_p` and `presence_penalty` continue to come from the
   model card and must still cite it.
2. A family with no readable card still gets **no** sampling keys, so its
   packaged defaults apply. Unchanged from ADR 0023.
3. `THINK_TEMPERATURE` is environment-overridable. `THINK_TEMPERATURE=1.0`
   reproduces a pre-2026-08-17 card-sampling run; `=0` is strict greedy.
4. `sampling_source` records the value — `card:gemma4+temp0.01`. A **bare**
   `card:<fam>` in an archived score identifies a run measured at the card's
   temperature 1.0, i.e. before this decision.
5. **Think-off is untouched.** It was greedy, it stays greedy, and every
   published think-off number, preflight expectation and release record remains
   valid.

## Consequences

- **This costs terminated cells, and the cost is already measured.** The
  negative-result document is the reference: nine models, three repeats, the
  eight `bbox_contract` arms. `gemma4:12b-it-q4_K_M` and
  `qwen3.6:35b-a3b-q4_K_M` lose **six of eight** cells at `0.01`;
  `gemma4:31b-nvfp4` loses 0–1 and `gemma4:31b-it-q4_K_M` loses none.
- **A `no result` cell under this policy is expected behaviour for the affected
  families, not a vision failure and not a regression.** Check the
  negative-result document before reporting one. This is the same class of trap
  as ADR 0022's `eval_count == num_predict`: a harness configuration that
  produces an empty response which reads as a model defect.
- **Where a cell scores, the score is good.** Measured IoU 0.94–0.97 on the
  cells that terminate. Low temperature does not degrade the answer, it prevents
  the answer — so this decision trades coverage for reproducibility, not quality
  for reproducibility.
- **Think-on numbers produced after this change are not comparable to the
  published card-sampling campaigns**: the 2026-08-14 on-policy runs (Apple and
  ROCm), the 2026-08-17 eighteen-model campaign, and every think-on row derived
  from them. Those remain valid at their own stated sampling; they are a
  different arm now, and `sampling_source` distinguishes them without ambiguity.
- Re-measuring the cost on a new family is one environment variable, not a code
  change.

## Alternatives considered

- **Keep card sampling and raise `n`.** Rejected on cost. The variance is large
  enough on categorical metrics that separating a real effect from noise needs
  far more repeats than a 12-test suite over multi-GB vision models can afford;
  the think-on half of a single two-model arm already runs to hours.
- **Keep card sampling and report variance instead of point values.** Honest,
  and rejected only for the gate: `expectations.toml` asserts values, and a
  preflight check cannot fail on a distribution without a decision rule that
  would itself need calibrating at high `n`. Campaign documents may still
  report spread.
- **Use `temperature 0`.** Rejected: strictly worse on the one axis that
  motivated ADR 0023. `0.01` is already greedy in all but name and already loses
  cells; `0` loses more, for no measured gain in reproducibility over `0.01`.
  Available via `THINK_TEMPERATURE=0` for anyone who wants to measure it.
- **Per-family temperature — low where it terminates, card where it does not.**
  Rejected: it reintroduces the comparability problem it was meant to solve, in
  the worst possible form, since the families would then differ in sampling
  regime as well as in the thing under test.
