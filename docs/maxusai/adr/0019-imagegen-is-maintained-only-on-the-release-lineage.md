# ADR 0019: MLX image generation is maintained only on `release/imagegen-mlx`; `main` returns to following upstream

- **Status:** accepted (2026-08-11)
- **Date:** 2026-08-11
- **Deciders:** MaxusAI fork maintainers
- **Related:** [ADR 0006](0006-release-lineage-is-never-merged-into-main.md) (the
  pattern this follows: a maintained lineage that is never merged into `main`),
  [ADR 0018](0018-imagegen-caches-mlx-streams-per-thread.md) (the imagegen fix
  this lineage preserves)

> **This ADR lives only on `release/imagegen-mlx`.** It is deliberately not on
> `main`, because it records why `main` no longer carries the subsystem.

## Context

Upstream removed the entire MLX image generation tree on 2026-07-28 in
`4713800b` (ollama/ollama#16615) — "Remove the x/imagegen tree (MLX image
generation engine, Flux2/zimage models, cache, C bindings) and all imagegen
integration points", including the server routes, the API surface and docs, the
OpenAI middleware image endpoints, the integration suites, and the imagegen path
in `x/create`.

The fork never took that commit. Measured 2026-08-11, `main` is **205 ahead and
43 behind** `upstream/main`, and the removal sits in those 43. Upstream's
`x/imagegen/` now contains only `manifest`; the fork carries the whole engine
(`mlx`, `nn`, `cmd`, `models`, `vae`, `tokenizer`, `cache`, …).

Until now there was **no recorded decision either way**. The tree was retained by
inertia rather than intent, which is precisely the situation ADR 0006 was written
to prevent for the other lineage. It also just received real work: `06a8437a`
(ADR 0018) fixed a long-standing SIGSEGV, and `x/imagegen` passes its tests for
the first time in this repo's recent history.

What forced the decision is the cost of *staying* diverged. Of the 43 upstream
commits the fork is missing, **21 touch `x/mlxrunner`** — the fork's core value
area — and only 1 touches `x/imagegen`. Continuing to carry a subsystem upstream
has abandoned makes every future upstream reconciliation wider, for a component
upstream will no longer fix, review, or test.

## Decision

1. **`release/imagegen-mlx` is a maintained lineage** carrying the MLX image
   generation engine, branched from `main` at `a8a25886` — the merge of PR #71,
   so it includes ADR 0018's thread-local stream fix and a green `x/imagegen`.
2. **`main` returns to following `upstream/main`**, which means taking
   ollama/ollama#16615 and dropping `x/imagegen` from `main`.
3. **Following ADR 0006, this lineage is never merged into `main`.** Fixes flow
   one way: `main` → lineage, cherry-picked when they apply.
4. This ADR is committed **only to this lineage**, per the same rule.

## Alternatives considered

- **Keep imagegen on `main`.** The status quo. Rejected because it is an
  unrecorded divergence that widens every upstream merge, and because the fork
  would own sole maintenance of an engine upstream has stopped shipping — a cost
  paid on `main`, where the fork's actual product (the MLX text runner) lives.
- **Delete imagegen outright and keep no lineage.** Simplest, and honest about
  where attention goes. Rejected because the engine currently works: the tree is
  wired into `runner.go`'s dispatch, has fork ADR coverage (ADR 0015), and was
  just repaired. Discarding a working subsystem the same week it was fixed throws
  away recoverable value for no gain over branching it.
- **Vendor imagegen into a separate repository.** Cleanest long-term separation,
  but it needs its own build, CI and release story for a component with no active
  consumer today. A branch costs nothing until that changes.

## Consequences

- Positive: `main` stops diverging on a subsystem upstream deleted, which makes
  the reconciliation of the 21 upstream `x/mlxrunner` commits the *only* hard
  part of catching up rather than one of two.
- Positive: the working engine, ADR 0018, and its regression test survive on a
  named branch instead of being lost.
- Negative: **`main` will carry no record of why `x/imagegen` disappeared**, since
  this ADR lives only here. Anyone reading `main` sees a subsystem vanish in an
  upstream merge with no fork-side rationale. Accepted deliberately; the pointer
  is this branch's existence.
- Negative: ADR 0018 remains on `main` describing code `main` no longer has, until
  the upstream merge lands and removes it there too.
- Negative: this lineage inherits the usual cost — it ages against `main`, and any
  MLX-layer fix that also applies here has to be cherry-picked.

## Conformance

- Branch `release/imagegen-mlx` exists at or after `a8a25886` and contains
  `x/imagegen/` with `x/imagegen/mlx/mlx.go` declaring
  `static __thread mlx_stream _default_stream`.
- `go test -p 1 -count=1 ./x/imagegen/...` is green on this lineage.
- `main` does not contain this ADR.
