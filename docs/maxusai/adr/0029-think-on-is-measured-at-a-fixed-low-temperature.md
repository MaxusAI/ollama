# ADR 0029: think-on is measured greedily at a fixed temperature, not at card sampling

- **Status:** accepted 2026-08-17. `THINK_TEMPERATURE` default is **0**;
  the `0.01` variant was measured and rejected (see below). Supersedes the *sampling* half of
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

It also made think-on **needlessly noisy**. At `temperature 1.0` two runs of the
same model over the same fixture differ, and this suite can afford `n <= 3`. At
that sample size a difference between two engines, two quantisations or two code
revisions cannot be separated from sampling noise. (The original framing here was
that card sampling made think-on "irreproducible" and that a fixed temperature
would restore reproducibility. The first half stands; the second does not — see
"Bit-reproducibility is unreachable" below. The goal is to remove the variance we
control, not to reach zero.) The
[low-temperature arm](../vision-lowtemp-thinkon-negative-result.md) was run to
test a fix and recorded as a negative result; the arm that motivated it —
`contract_followed` at `n=3` — showed the problem directly, with per-repeat
values like `6/8/8` and `7/5/6` on a categorical metric.

The suite exists to gate regressions. A regression gate needs a number whose
movement means something — which requires the noise under it to be smaller than
the effect being gated, not zero. On-policy realism and low variance are in
direct conflict here, and only one of them is load-bearing for that job.

## 0.01 was tried first, and it does not deliver the thing this ADR is for

The first draft of this decision set the default to `0.01` — near-greedy, on the
reasoning that it would be reproducible without being greedy. **It is not
reproducible.** Two runs of `gemma4:31b-nvfp4` think-on at `0.01`, same fixture,
same window, cold server both times, `scene_single` first in each:

| run | `eval_count` | scene IoU |
|---|---|---|
| A | 3205 | 0.965 |
| B | 1762 | 0.961 |

Nearly double the reasoning tokens, and the scored IoU moved 0.004 with it.

At the time this was read as fatal — determinism refuted, therefore worthless.
That reading was wrong, and the section below is why: `temperature 0` is not
bit-reproducible either, so refuting determinism does not distinguish the two.
What distinguishes them is measured spread on the scored surface, and there `0`
wins: `scene_single` is identical across three repeats at `0` and moved 0.965 ->
0.961 at `0.01`.

Recorded rather than quietly dropped, because `0.01` is the intuitive choice and
the next person will reach for it too — and because the *reasoning* that first
rejected it was itself faulty, which is worth more to a future reader than the
verdict.

## Decision

**Think-on is measured at `THINK_TEMPERATURE`, default `0` — greedy — for every
family. Temperature is fork policy. Every other sampling field stays
card-sourced.**

1. `sampling.py` supplies `temperature` from `THINK_TEMPERATURE`;
   `top_p`, `top_k`, `min_p` and `presence_penalty` continue to come from the
   model card and must still cite it.
2. A family with no readable card still gets **no** sampling keys, so its
   packaged defaults apply. Unchanged from ADR 0023.
3. `THINK_TEMPERATURE` is environment-overridable, and stays so for one reason:
   `THINK_TEMPERATURE=1.0` is the only way to reproduce the published
   card-sampling campaigns. Removing the knob would make those numbers
   unreproducible rather than merely superseded. The **default** is what governs
   new work.
4. `sampling_source` records the value — `card:gemma4+temp0`. A **bare**
   `card:<fam>` in an archived score identifies a run measured at the card's
   temperature 1.0, i.e. before this decision.
5. **Think-off is untouched.** It was greedy, it stays greedy, and every
   published think-off number, preflight expectation and release record remains
   valid.

## The runaway risk did not materialise on the model measured

This walks deliberately back into greedy decoding, which is the defect ADR 0023
was written to escape, so the first thing measured was whether reasoning still
terminates. On `gemma4:31b-nvfp4`, MLX/CUDA, think-on at `temperature 0`, the
full suite with the ladder live (start rung 16384, `num_predict` 8192 derived):

- **Every cell terminated. None capped, none escalated.** Worst cell 3697 of
  8192; the `bbox_contract` family sat between 764 and 2025.
- **16384 is therefore the window this model needs for a thinking response.**
  That is a measured throughput fact, not an assumption, and it is only
  obtainable because the ladder was allowed to climb — see SPEC H4a.

This does **not** generalise. It is one model on one stack, and it is precisely
the model the negative-result document already identified as the robust one.
`gemma4:12b-it-q4_K_M` and `qwen3.6:35b-a3b-q4_K_M` lost six of eight cells at
`0.01`, and `0` is stricter.

## Bit-reproducibility is unreachable, and it was the wrong target

The premise this ADR was first drafted on — pick a temperature and think-on
becomes deterministic — is **false, and not only for `0.01`.** Measured `n=3`,
`gemma4:31b-nvfp4`, MLX/CUDA, think-on at `temperature 0`:

| axis | result |
|---|---|
| `eval_count` | **3 of 12 cells stable.** `document_single` ran 3697 / 1982 / 2398 |
| scored fields | **6 of 12 byte-identical across all three** |

Greedy has already removed sampling as a source of variance, so what remains is
the platform: GPU reduction order varies with scheduling, argmax tie-breaks
differently, and one divergent token sends the reasoning trace down another
path. Stated as the hypothesis it is — it has not been isolated — but the
consequence is not hypothetical: **no temperature setting will make think-on
bit-reproducible on this stack**, and chasing one is chasing the wrong quantity.

**What is stable is the thing a gate actually asserts.** Of the six scored cells
that vary, five vary only in `iou_declared` by 0.001–0.005 — every one inside
ADR 0012's ±0.01 noise floor, which exists for precisely this:

    bbox_contract_multi       0.966 / 0.966 / 0.965      spread 0.001
    bbox_contract_anchored    0.966 / 0.967 / 0.967      spread 0.001
    bbox_contract_adv_norm1   0.965 / 0.965 / 0.964      spread 0.001
    bbox_contract_perobject   0.960 / 0.960 / 0.963      spread 0.003
    bbox_contract_reasoning   0.961 / 0.966 / 0.961      spread 0.005

The sixth is categorical and real: `document_single` gives `name_bbox_hits`
5 / 4 / 4, with `name_bbox_space` flipping between `pixel/xyxy` and
`norm1000/xyxy`. That is the cell the ROCm campaign already recorded swinging
0.638 -> 0.248 across configurations — the suite's least trustworthy metric,
behaving as documented.

**So `0` is still the right choice over `0.01`, on evidence rather than on
principle:** `scene_single` scores identically across all three repeats at `0`
(IoU 0.966 x3) where `0.01` produced 0.965 and 0.961. Removing the last sampling
noise tightens the scored surface even though it cannot make it exact.

**What this forbids.** A think-on `expectations.toml` assertion on a
**categorical** metric at `n=1` is not admissible — `name_bbox_hits` alone would
flip a gate two runs in three. Continuous metrics are admissible against the
±0.01 floor. Think-on campaign tables must report the rung and should report
spread, not a point value, on any cell outside the `bbox_contract` family.

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
- **Use `temperature 0.01`.** This was the first draft's decision, rejected on
  measured spread rather than on determinism (neither value is deterministic):
  `scene_single` is identical across three repeats at `0` and moved 0.965 ->
  0.961 at `0.01`. Its one advantage over `0` — marginally less non-termination
  pressure — is unquantified, and on the model measured here `0` did not lose a
  single cell, cap a single cell, or need a rung above 16384.
- **Per-family temperature — low where it terminates, card where it does not.**
  Rejected: it reintroduces the comparability problem it was meant to solve, in
  the worst possible form, since the families would then differ in sampling
  regime as well as in the thing under test.
