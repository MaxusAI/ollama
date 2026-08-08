# ADR 0011: pre-deploy expectations are versioned code in the repo, not skill knowledge

- **Status:** accepted, 2026-08-08
- **Date:** 2026-08-08
- **Deciders:** MaxusAI fork maintainers
- **Related:** [ADR 0008](0008-gemma4-budget-fill-restores-1120.md) and
  [`005-llama-cpp-dynres-pinned-overshoot.patch`](../../../llama/compat/005-llama-cpp-dynres-pinned-overshoot.patch)
  (both changed expected values inside one week),
  [ADR 0006](0006-release-lineage-is-never-merged-into-main.md) (why one baseline
  cannot cover every host),
  [AMD upgrade gate](../amd-upgrade-gate.md) (the ROCm host runs a different payload),
  [ADR 0003](0003-vision-image-token-budget-policy.md) (budgets are per-arch)

## Context

Every new image build was verified by re-running the same checks by hand, and the
checks drifted. GitHub Actions cannot run the matrix: it spans CUDA, ROCm and
Apple Silicon — partly shared assertions, partly platform-specific — on hosts with
real GPUs, and this fork has no self-hosted runners registered at all.

That is a limit on running the *probes*, not on enforcing the *data*. The
consistency of `expectations.toml` needs no GPU and no server, and is enforced in
CI — see the last consequence below.

The vision suite already had the *probes*
([`vision_suite.py`](../vision-suite/vision_suite.py),
[`measure.py`](../vision-suite/measure.py),
[`extbench.py`](../vision-suite/extbench.py)) plus a no-GPU formula gate
(`go test ./llm/ -run TestImageTokensForSize`). What was missing was a **pass/fail
layer with expectations as data** — `measure.py`'s own README notes it *reports
rather than asserts*.

The obvious shortcut — put the checks and their expected values in an agent skill —
fails for one reason: **expected values are a property of the payload, and the
payload changes.**

## Evidence

Two expectation changes landed in a single week, both correct, both from compat
patches:

| patch | what changed | expected value before → after |
|---|---|---|
| [004](../../../llama/compat/004-llama-cpp-gemma4-budget-fill.patch) | gemma4 budget-fill sizing ([ADR 0008](0008-gemma4-budget-fill-restores-1120.md)) | size-scaling ladder → **flat 1102 at every resolution** |
| [005](../../../llama/compat/005-llama-cpp-dynres-pinned-overshoot.patch) | pinned dyn_size budgets clamp to the ceiling | nemotron pinned 3328 delivered **3390 → 3270** |

Anything holding those numbers outside the repo rots silently against them, and
nothing tells you it has.

The split is also load-bearing in the other direction. A verdict rule that reads
the *shape* of a result without knowing the arch gets 004 exactly backwards: a flat
token ladder means an **unpatched payload** for `nemotron_h_omni`, but is the
**correct** result for `gemma4`. That inversion was hit twice by hand before the
rule was written down as data.

Measured 2026-08-08 against the reference canary
(`maxusai/ollama:4987dd49-dynres`, `0.32.5-dynres-4987dd49`, payload 001+002+004+005):
13 checks pass, 1 skipped deliberately. The ladder reproduces the recorded values
exactly on both arches — nemotron **266/266/578/2306/3270**, gemma4 flat 1102 — and
the pinned probe reads 3270 against a 3328 ceiling.

**Getting the subtrahend right took two corrections, and the first one alone looked
sufficient.** Both are worth recording, because each produced a convincing false
regression before it was understood:

1. **Mismatched prompts.** `measure.py` baselines with `"Hi"` but probes images with
   `"Describe briefly."`, so its deltas carry the difference in *prompt* length —
   18 vs 21 tokens on `nemotron3:33b-q8`, a constant +3 on every row. Inheriting
   that made all five ladder rows read +3 and look like a payload regression.
2. **The text prefix is not the text-only count.** Matching the prompt does not
   finish the job: attaching an image changes how the template renders the
   surrounding text. Same prompt, same model — text-only 21, prefix inside an image
   request 20 — so a matched-prompt text-only baseline reads every nemotron image
   exactly 1 token *low*. `gemma4:31b` measures 19 both ways, so the offset is
   arch-specific and cannot be hardcoded.

Correction (1) alone is why this ADR originally recorded the ladder as
`265/265/577/2305/3269`. That row was uniformly 1 low, and the discrepancy was
briefly dismissed as grid quantisation — wrongly, since quantisation does not land
deterministically one low on every rung. The internal contradiction was visible at
the time: the ladder's 2048×1152 entry read 2305 while the pinned check's unpinned
control read 2306 for the same geometry. The harness now recovers the prefix from a
two-image difference, `count(A) + count(B) − count(A,B)`, which cancels it without
trusting a text-only probe or assuming a grid. Normative in
[SPEC B8](../spec/vision-image-token-budgets.md).

## Decision

**Expected values are versioned code in this repo. The skill owns procedure only.**

Normative rules:

1. **Expectations live in
   [`docs/maxusai/vision-suite/preflight/expectations.toml`](../vision-suite/preflight/expectations.toml)**,
   keyed by `(platform, payload patch set)` → arch. They are versioned alongside
   the payload they describe and change in the same commit as the patch that moves
   them.
2. **The [`ollama-preflight` skill](../../../.claude/skills/ollama-preflight/SKILL.md)
   contains no expected values** — no token counts, no thresholds. It knows which
   host, which order, how to detach, how to read a verdict. Code owns the
   assertions; the skill owns the procedure.
3. **Verdicts are per-arch.** Each arch declares `scaling = "dynamic" | "flat"` and
   the diagnosis follows from it. A shared shape heuristic is forbidden — it
   inverts under 004.
4. **An unmeasured expectation is never a pass.** A `(platform, arch)` combination
   with no measured baseline reports `NEEDS_BASELINE` and exits 4. Copying another
   profile's numbers across to make a run green is the failure this file exists to
   prevent.
5. **An unknown `(platform, version)` combination is a hard error (exit 2), never a
   default.** One baseline does not fit all hosts. The ROCm/gfx1151 host runs
   `0.32.1-dynres` from the release lineage — **the same compat patch set as CUDA
   (001+002+004+005) over a different llama.cpp payload, b9888 rather than b10091**.
   The [AMD gate](../amd-upgrade-gate.md) blocks the 0.32.5 base and its b10091
   payload, not the compat patches, which reached that lineage as adapted backports
   (`5f6e7fdc`, `593fc3b1`, `35d9e58e`). Identical patch lists therefore do **not**
   imply identical numbers, which is why profiles key on `(platform, version)` and
   not on the patch set. A 0.32.5-dynres build declared as `--platform rocm` is
   refused rather than validated against the wrong baseline.
6. **The payload patch proof is the model-load log, never binary inspection.**
   `load_hparams: image_max_pixels: N (custom value)` with `N == max_tokens × S²`,
   `S = patch_size × n_merge`. Static inspection of `libmtmd.so` is unreliable and
   must not be reintroduced: `strings` is absent from the ollama images so
   in-container greps return misleading zeros, and `<img>`/`</img>` literals appear
   in stock too (InternVL uses them). The harness forces a model unload first so it
   cannot read a previous build's line.
7. **The version string is asserted before any measurement is trusted**, and a
   mismatch aborts the run. 11434/11435/11436 are all occupied on 10.8.0.6 and a
   canary once answered from the wrong server; only a mismatched version caught it.

## Consequences

- Adding a compat patch that moves a measured value is **not complete** until
  `expectations.toml` is updated in the same change. The procedure is
  [preflight/README.md](../vision-suite/preflight/README.md), "Adding an
  expectation".
- `preflight/test_verdicts.py` guards the guard: it asserts the verdict logic
  (flat-vs-dynamic both ways, the pinned ceiling invariant, the `num_predict` trap)
  and the internal consistency of the data file itself — every `image_max_pixels`
  equals `max_tokens × S²`, every `scaling` field agrees with the ladder beside it,
  every `unmeasured` block carries a reason. Editing a ladder to flat without
  updating `scaling` fails there, in milliseconds, with no GPU.
- **That guard runs in CI**, via `.github/workflows/preflight-expectations.yaml`.
  It is a separate workflow because `test.yaml` excludes `docs/**`, so a pull
  request touching only `preflight/` otherwise runs no checks at all — and a
  docs-only diff is exactly what "adding an expectation" looks like. It triggers
  unconditionally and scopes itself internally rather than filtering on paths,
  because a path-filtered workflow can never be a *required* check: GitHub waits
  forever for a report that never comes on unrelated pull requests.
- The ROCm and Apple Silicon profiles ship **declared but unmeasured**. Those hosts
  exit 4 until someone measures them. That is the intended state, not a gap to paper
  over.
- Per [ADR 0006](0006-release-lineage-is-never-merged-into-main.md) rule 1, the
  harness reaches `release/0.32.1-dynres` by **cherry-pick from `main`**, never by
  merge. Until it is cherry-picked, the ROCm profile has no branch to run on — the
  gfx1151 host is served from the release lineage.
- Results are machine-readable JSON with a `meta` block naming host, platform,
  profile, patchset and version, so runs are aggregated by hand across hosts until
  CI exists. Report per-host; a merged pass/fail hides that ROCm is on a different
  payload by design.
