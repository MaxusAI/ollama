# ADR 0033: Adopt upstream's MLX grammar engine; supersede the fork's pure-Go constrained sampling

- **Status:** accepted (2026-08-28). Supersedes
  [ADR 0009](0009-mlx-pure-go-constrained-sampling.md) as the *implementation*;
  the guarantee ADR 0009 exists to protect is retained and re-tested.
- **Date:** 2026-08-28
- **Deciders:** Glenn; assessed in the v0.33.2 sync (task
  `upstream-sync-2026-08-28.md`)

## Context

ADR 0009 gave the MLX runner a pure-Go JSON grammar (`x/structured`), wired in
through `Request.Constraint` / `compileFormat`, with a masked constrained
decoder and, later, grammar-aware speculation.

Upstream v0.33.2 (`147509c0`, "mlxrunner: add structured output support")
implements structured output in the same place with a different design:
`grammarEngine.prepare()` compiles **asynchronously**, overlapping tokenization
and prefill, into `Request.Grammar`, and speculation carries the grammar
through the parked inner decoder — "a constrained session never drafts".

The two collide across `pipeline.go`, `speculate.go` and `runner.go`. One has
to go.

## Decision

**Adopt upstream's engine.** The fork's constraint layer is retired.

Three things decided it:

1. **The fork's speculation advantage was never realised.** PR #201 measured it:
   correct, but **inert** — `drafted=0`, because a cold-start deadlock in the
   depth controller means a round never proposes. Three candidate fixes were
   written down and none was picked. We are trading away an unrealised gain.
2. **Upstream reaches the same effective behaviour** — no drafting under
   constraint — deliberately rather than by deadlock, and adds an async compile
   we did not have.
3. **Upstream is now actively developing this surface.** Carrying a competing
   implementation here means re-resolving this collision at every sync.

## The guarantee is retained

ADR 0009 exists to protect one contract: *a format the runner cannot honour is
an **error**, never a silently dropped constraint.* Upstream's `parseGrammar`
implements exactly that, and is stricter than ours was:

| format | ADR 0009 (`compileFormat`) | upstream (`parseGrammar`) |
|---|---|---|
| absent / `null` / `""` | unconstrained | unconstrained |
| `"json"` | constrained | constrained |
| `"yaml"` | **error** | **error** |
| malformed schema | error | error, plus size limit and UTF-8 validation |

`client_format_test.go` no longer tests `compileFormat`; it asserts the same
table against `parseGrammar`, so the contract stays covered by a test that
fails if a future change starts dropping constraints silently.

The raw-GBNF rejection ADR 0009 also specified is now **structural**: upstream
deleted `CompletionRequest.Grammar` (`7027546c`), so a caller can no longer
express a raw grammar to reject.

## Consequences

- `x/structured` is no longer on the MLX request path. It is still a tested
  package and still used elsewhere; it is not deleted here.
- **`constrain.go` (478 lines) is now unreachable, not half-wired.** `s.matcher`
  is set only by `attachGrammar`, which after this change is called from
  nowhere; every masking path guards on `s.matcher == nil` and `maskRows`
  returns its input when there are no masks. Verified by inspection, and the
  package's tests pass. **Follow-up: delete it and its `speculate.go` call
  sites** — deliberately not done inside this merge, to keep the diff
  reviewable.
- Grammar-aware speculation (#191, #201) is retired with it. If the depth
  controller is ever fixed, it would have to be rebuilt on upstream's engine.
- **Not runtime-validated.** This lands on `go build` plus a green unit suite.
  The MLX path needs a structured-output request against a built image before
  deploy; unit tests do not exercise MLX kernels.
