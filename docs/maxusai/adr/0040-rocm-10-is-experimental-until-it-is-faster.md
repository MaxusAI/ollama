# ADR 0040: ROCm 10.0.0 is carried as experimental, not promoted

- **Status:** accepted 2026-09-21 (Glenn: "Keep the ROCm 10 as in experimental state").
  Sits on [ADR 0011](0011-preflight-expectations-are-versioned-code.md) for why it has no
  measured profile, and on the AMD upgrade gate
  ([amd-upgrade-gate.md](../amd-upgrade-gate.md), the 2026-09-21 decision) for the
  promotion it was weighed against.
- **Date:** 2026-09-21
- **Deciders:** MaxusAI fork maintainers

## Context

ROCm 10.0.0 was built, executed and measured on gfx1151 for the first time in this cycle.
The fork had been on 7.2.4; 7.14.1 was the original upgrade target and aged out of every AMD
package channel while the work was in progress, leaving 10.0.0 as the only obtainable release.

Every soundness question asked of it has been answered, and all of them affirmatively:

| question | answer |
|---|---|
| Does it run on gfx1151? | Yes — `devices=1`, 61/61 layers offloaded, tokens generated |
| Do the other 12 preset architectures compile under clang 23? | Yes — all 13, device code verified **in the binary**, not inferred from an exit code |
| Gate clause 3 (dio load-path integrity)? | **0/54** blocks differ, arms verified to differ at the runner flag line |
| Gate clause 4 (vision A/B, 0 degenerate)? | Passed, five models, **zero degenerate rows** |
| Preflight? | **PASS 15/0/14**, all three token ladders unmoved against 7.2.4's measured values |
| The two moved fine-text cells? | Identical to 7.2.4, N=5, zero within-arm variance |

It is not a broken build. It is a working one that loses the only race that matters here.

## Decision

**ROCm 10.0.0 is carried as an experimental surface. Production stays on 7.2.4.**

Three things follow, and they are the substance of this ADR rather than the headline:

1. **`rocm10` is a real platform in the preflight harness, with no measured profile.** The
   `rocm` platform split into `rocm7`/`rocm10` because the ollama version string does not
   encode the ROCm release: a 7.2.4 build and a 10.0.0 build of the same fold both stamp
   `0.34.2-dynres-<sha>`, and `resolve_profile` returns the first hit, so two profiles on one
   platform would have been resolved by **dict order, silently** — with the losing profile's
   `toolchain_build` pin never firing. The surface is listed so it reads *not run* rather than
   vanishing, because a missing row reads as "not applicable", which is the more dangerous of
   the two.
2. **No profile is fabricated to make it look ready.** A measured profile requires a build
   stamped `0.34.2-dynres-<sha>`; the only ROCm 10 image that exists is
   `0.34.2-dynres-rocm10probe`, from an unmerged branch, and `rocm10probe` is not a hex SHA.
   Writing a `version_pattern` that accepted it would encode a throwaway build name as if it
   were a release line. ADR 0011 rule 4 forbids exactly this.
3. **A standing instance runs on `:11500`** (`ollama-rocm10`), sharing the production model
   store, so experimental means *available for testing*, not *shelved*. It carries
   `OLLAMA_NOPRUNE=1` and `OLLAMA_MAX_LOADED_MODELS=1` — not tuning: the first so it can never
   prune blobs the serving instance depends on, the second because this is an iGPU sharing
   system RAM with the instance actually serving requests.

## Why not promote it

**It is slower where this deployment is bound.** Against a ±0.4% noise floor measured on this
host from two direct-I/O pairs — same image, same ROCm, a knob with no score effect — prefill
on blocks that actually encoded is down on **all five** models:

| model | Δ gen | Δ prefill (cold) |
|---|---|---|
| `qwen3.6:35b-a3b` | +4.6% | **−6.1%** |
| `qwen3.8:27b` | +1.6% | **−3.3%** |
| `gemma4:31b` | +6.4% | **−5.5%** |
| `gemma4:26b-a4b` | +4.2% | **−10.4%** |
| `nemotron3:33b` | +0.5% | **−8.8%** |

Prefill is the image-encode path, which is what vision work waits on. A few percent of decode
does not pay for it. The regression is flat across prompt sizes within a model and varies by
model, so it tracks architecture rather than image size — consistent with kernel selection,
though not measured to it.

It also changes **7 of 135 scored blocks** against a control that changes **0 of 108**, so the
movement is real ROCm-induced numerics rather than sampling. The direction is 5 better / 2 worse
on borderline `contract_followed` items, which is a coin toss, not an improvement.

And it exists only on an unmerged branch, so promoting it would have made production
non-reproducible from `main`.

## Options considered

- **Promote it.** Rejected on the measurement above. Soundness is not superiority, and this is
  the distinction the whole cycle turned on.
- **Drop the work.** Rejected. The validation is expensive and perishable, and 7.2.4 will not
  be obtainable forever — AMD's channels already carry only 10.0.0. When that upgrade becomes
  forced rather than chosen, this evidence is what makes it a day's work instead of a month's.
- **Merge the build path but ship nothing.** This is effectively what was chosen; the
  distinction is that the surface is *declared* in the harness, so its absence of measurement
  is visible rather than implicit.

## Consequences

- The release matrix shows `rocm10` as **not run**, permanently, until someone measures it.
  That is the intended reading, not a gap to be closed by writing rows.
- Two ROCm platforms must be kept apart by name forever, because the version string will never
  distinguish them. `test_no_two_profiles_share_a_platform_and_pattern` enforces it.
- The 7.14.1 target is dead: no tarball, RPM or wheel is obtainable from any AMD channel, and
  it exists only as three Docker tags.
- **What would reverse this:** a ROCm release that is not slower at prefill on this hardware,
  or 7.2.4 becoming unobtainable or unsupportable. Either one, and the path is already walked —
  merge the Dockerfile, rebuild with a real stamp, commit the measured profile, promote.
