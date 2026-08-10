# ADR 0016: The scheduler reloads on resolved vision flags, not on raw budget options

- **Status:** accepted, implemented + verified 2026-08-10 on fork `main`
  lineage; backported here because this lineage carries the same
  `nemotronImageTokenBudget` sentinel and the same `needsReload` comparison.
  Refines invariant B3 of
  [SPEC: vision image token budgets](../spec/vision-image-token-budgets.md).
- **Date:** 2026-08-10
- **Deciders:** MaxusAI fork maintainers

## Context

`ImageMinTokens`/`ImageMaxTokens` are `api.Runner` options — shared across
every arch, but carrying *gemma4's* defaults (main ADR 0008, backported here
as compat/004), because `api.DefaultOptions()` has one set of values for all
models. Arches with their own native bounds therefore treat the gemma4
defaults as "caller left this alone" and substitute their own (main ADR 0001
for nemotron dynres; the sentinel is the `api.Default*` constants, so its
literal value moves with the gemma4 default — 70/1120 today).

The consequence was a reload bug. `runnerRef.needsReload` compared the raw
`api.Runner` structs with `reflect.DeepEqual`, so on `nemotron_h_omni` an
unset request (carrying 70/1120) and one explicitly asking for 256/3328 — the
arch's own bounds, the identical `--image-min-tokens`/`--image-max-tokens`
launch — compared unequal and evicted a loaded model for nothing. The same
held for arches whose vision flags ignore these options entirely: the qwen
family passes a fixed floor, and most arches pass no budget flags at all, yet
any change to the bounds forced a reload that could not change the launch.

## Decision

Compare what the runner would be **launched with**, not what the caller
happened to send.

`llm.ResolvedImageTokenBudget(modelArch, opts)` reports the bounds
`visionServerArgs` will pass for an arch and whether that arch derives them
from the options at all; `llm.NormalizeImageTokenBudget` rewrites an
`api.Runner` to those values, zeroing both where they cannot affect the
launch. `needsReload` applies it to each side before the `DeepEqual`, beside
the existing `NumCtx`/`NumBatch`/`UseMMap`/`NumGPU` normalizations.

Enforced by `TestSchedNeedsReloadImageTokenBudget` (`server/sched_test.go`),
which pins both directions — the sentinel-equivalent and flag-ignoring cases
must not reload, a genuinely different ceiling must — and
`TestResolvedImageTokenBudget` (`llm/llama_server_test.go`).

## Options considered

- **Normalize to resolved flags before comparing** (chosen) — fixes the
  reload for every arch at one seam, needs no wire or type change, and keeps
  the "budget changes reload the runner" contract intact for changes that
  actually alter the launch.
- **Make the bounds `*int` so "unset" is representable** — the clean fix for
  the *underlying* ambiguity, and consistent with `MainGPU *int` /
  `UseMMap *bool` in the same struct. **Rejected by the maintainer** on
  2026-08-10 as public-API churn disproportionate to the benefit, once the
  reload cost — the part users actually feel — was addressed separately. Note
  it would not have fixed the reload on its own: `DeepEqual(nil, &256)` is
  still unequal, so resolved-flag comparison is required either way.
- **Special-case the sentinel inside `needsReload`** — a per-arch `if` at the
  scheduler, duplicating knowledge that already lives in the resolvers and
  silently rotting the next time an arch gains a budget.

## Consequences

- A client that names an arch's own bounds explicitly, or that varies the
  budget on an arch which ignores it, keeps the loaded runner. Changing the
  effective budget still reloads, as B3 requires.
- The sentinel ambiguity itself remains: an explicit `70/1120` on nemotron is
  still read as unset and is not expressible. That is now a documented
  limitation rather than an accident — see the resolver's comment.
- `ResolvedImageTokenBudget` is the single place that answers "what budget
  will this arch launch with", so an arch added to `visionServerArgs` must be
  added there too or its budget silently stops forcing reloads. The two
  switches sit in the same file for that reason.
- Lineage note: this backport carries only the arch-specific half (main ADR
  0003). `Gemma4ImageBudget` and the MLX-runner media paths that motivated
  main's neighbouring commits are deliberately absent here.
