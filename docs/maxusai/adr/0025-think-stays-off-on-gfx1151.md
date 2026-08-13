# ADR 0025: on gfx1151/ROCm, think stays off for every measured family

- **Status:** accepted (2026-08-14)
- **Date:** 2026-08-14
- **Deciders:** MaxusAI fork maintainers
- **⚠ Lineage note.** Backported from `main` (`40da2b9c`). Unlike the ADR 0023 backport,
  **this one was measured on this lineage**: the campaign ran against
  `0.32.1-dynres-296eb020` — the artifact this branch produces — on gfx1151/ROCm, payload
  b9888. It replaces the "unmeasured rather than approved" caveat carried in ADR 0023's
  lineage note here. Adapted per ADR 0006: the campaign record and ADR 0012 live on `main`
  and are named rather than linked; the raw scores for all 12 cells are on `main` under
  `docs/maxusai/vision-suite/runs/onpolicy-rocm-2026-08-14/`.
- **Scope:** **host-scoped.** This ADR does not supersede
  [ADR 0023](0023-think-mode-is-per-model-and-measured-on-policy.md); it *applies* it, and
  supplies the ROCm reconfirmation 0023 deferred. ADR 0023 remains the governing framework —
  think mode is decided per model, on on-policy measurements — and its Apple Silicon verdicts
  stand for that platform.
- **Related:** `docs/maxusai/vision-campaign-2026-08-14-onpolicy-rocm.md` (on `main`) (the
  measurements this rests on), [ADR 0023](0023-think-mode-is-per-model-and-measured-on-policy.md)
  (the framework and the deferred reconfirmation),
  [ADR 0022](0022-thinking-is-off-for-vision-work.md) (superseded by 0023; its practical
  guidance for this host turns out to have been right for the wrong reason),
  ADR 0012 (on `main`) (±0.01 noise floor)

## Context

ADR 0023 retired ADR 0022's blanket `think: false` rule and replaced it with per-model
verdicts, on the strength of an on-policy campaign. Two of its three verdicts kept think off;
the third — **gemma4** — was reversed to *permitted*, on the finding that the grounding cost
ADR 0022 recorded was an artefact of greedy decoding.

ADR 0023 measured on Apple Silicon / b10353 and said so, forbidding any pooling with gfx1151
results and listing ROCm reconfirmation as deferred:

> that host is unreachable from here. The per-model decisions above are scoped to the measured
> host and should be reconfirmed there.

That reconfirmation has now been run on gfx1151, on-policy, n = 3, under the same harness and
the same admissibility rules.

## Evidence

Full data and cap audit in the the 2026-08-14 campaign record (on `main`).
Server `0.32.1-dynres-296eb020`, payload b9888, `/api/generate`, `num_ctx 32768`,
`num_predict 24000`. Deltas are within-host; nothing is compared across hosts except the
*sign* of each delta.

**gemma4:31b — the one verdict that does not transfer.**

| | document-IoU deltas, think-on vs think-off | mean |
|---|---|---|
| **gfx1151** | −0.062, −0.012, −0.010 | **−0.028** — negative in every rep |
| Apple Silicon (ADR 0023) | +0.001, −0.001, +0.047 | +0.016 — negative in none |

12/12 cells valid, none capped, coordinate dialect stable in every rep — the cleanest family
in the campaign, so the result is not an artefact of truncation or dialect flipping. Scene
adds one −0.071 rep against two at parity. Reasoning also costs **11–16×** the think-off
tokens here against ADR 0023's ≈4×, on the same model and the same card values.

**qwen3.6:35b-a3b — confirmed, worse.** Two of three `scene_single` cells hit `num_predict`
exactly with `json_valid: false`, so their IoU is inadmissible; the one rep that terminated
flipped the bbox dialect `norm1000` → `pixel`, reproducing ADR 0023's rep 2. Document IoU
regressed in all three (−0.030, −0.043, −0.147), and those cells did not cap.

**nemotron3:33b — confirmed, on ADR 0023's own terms.** Scene 0.840 → 0.736 / 0.165 / 0.384,
**no scene cell capped**, dialect stable in all three reps. Neither failure mode that sampling
explains is present, which is precisely the argument ADR 0023 makes for this family; it now
reproduces on different silicon, a different payload and a different sampling source.

## Decision

**On gfx1151/ROCm, vision is served with `think` off for all three measured families.**

1. **`gemma4` — think off on this platform.** ADR 0023's permission is scoped to its
   measurement and does not reproduce here. Do not enable thinking for gemma4 on gfx1151 to
   improve vision output; it costs document grounding in every measured rep. If a reasoning
   trace is itself the deliverable, it remains available at 11–16× the tokens, with that cost
   understood.
2. **`qwen3.6` — think off.** Unchanged from ADR 0023 and reinforced.
3. **`nemotron3` — think off.** Unchanged from ADR 0023 and reinforced.

**This is not a return to ADR 0022's blanket rule.** The rule here is per-model and
per-platform, and it happens to come out uniform on this host. Any new family, or any payload
change on this lineage, needs its own measurement — the outcome above is not a prior for
models that were not measured.

**ADR 0023's admissibility rule applies unchanged and was applied here:** `sampling_source`
present and not `legacy-greedy`, n ≥ 3 per think-on cell, and `eval_count` checked against
`num_predict` before any score is read. Four cells failed that last check and their affected
tests are excluded above rather than reported.

## Consequences

- **A per-model verdict does not survive a host change by default.** This is the concrete
  demonstration: three families measured under one framework on two platforms, and one
  verdict inverted. ADR 0023's instruction not to pool the hosts was not a formality. Any
  future think-mode decision must name the platform it was measured on.
- **The mechanism of a finding can be wrong while its guidance is right.** ADR 0022 told this
  host to keep think off, and that guidance survives — but its stated reasons (gemma4 losing
  0.04–0.09 document IoU to thinking as a general property; qwen3.6 failing on multi-image)
  were both wrong, and ADR 0023 corrected both. Being right about what to do is not evidence
  of being right about why.
- **gfx1151-specific observation:** nemotron3's `finetext` hit the 24 000 cap in 2 of 3 reps.
  ADR 0023 reports 12/12 valid with no caps for this family on its host, so this is an
  addition to the picture, not a contradiction of it.
- **qwen3.6 scene grounding on-policy is unmeasured on this host, not measured-bad.** Two of
  three cells never produced a scorable answer. Reporting those as 0.0 would be exactly the
  trap ADR 0022 identified and ADR 0023 carried forward.
- **Scope of the numbers.** One host, one payload (b9888, pinned by the AMD upgrade gate),
  gemma4 at 31B only. n = 3 establishes sign consistency, not a confidence interval.

## Alternatives considered

- **Adopt ADR 0023's gemma4 permission here on the strength of its Apple Silicon result.**
  Rejected: that is the pooling ADR 0023 itself forbids, and the measurement now shows the
  sign differs by host.
- **Reinstate ADR 0022's blanket rule.** Rejected: it reaches the right answer for this host
  by an argument that is known to be wrong, and it would prejudge families and platforms that
  have not been measured.
- **Raise `num_predict` until qwen3.6 `scene_single` terminates.** Not attempted: ADR 0023's
  fourth trap records that above ~90 K the 1800 s HTTP timeout expires first, converting a cap
  into an error with no data. The honest report is "unmeasured".
- **Re-run with more replicates.** Deferred, not rejected. n = 3 satisfies the admissibility
  rule and the gemma4 sign is consistent across all three; more replicates would tighten the
  deltas but not change the decision.
