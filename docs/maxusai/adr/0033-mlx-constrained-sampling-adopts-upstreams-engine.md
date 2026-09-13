# ADR 0033: Adopt upstream's MLX grammar engine; supersede the fork's pure-Go constrained sampling

- **Status:** accepted (2026-08-28); amended 2026-09-12 (upstream drafts under a grammar again; we keep it, with a
  knob). Supersedes
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
  *(2026-09-04: with `constrain.go` deleted it has no remaining in-tree
  importer. Kept deliberately, per this bullet.)*
- **`constrain.go` (478 lines) is now unreachable, not half-wired.** `s.matcher`
  is set only by `attachGrammar`, which after this change is called from
  nowhere; every masking path guards on `s.matcher == nil` and `maskRows`
  returns its input when there are no masks. Verified by inspection, and the
  package's tests pass. **Follow-up: delete it and its `speculate.go` call
  sites** — deliberately not done inside this merge, to keep the diff
  reviewable. **Done 2026-09-04:** `constrain.go` and its six test files are
  gone (1,303 lines), along with the inert call sites in `speculate.go`
  (`adoptGrammar`, `maskRows`, `verifyPlan`, `errNoLegalDraft`, the
  `verifyTokens`/`verifyDist` views and the session's matcher/vocab/pieces
  fields), `runner.go`'s `constraintVocab` cache, and the permanently-zero
  `grammar_truncated` / `grammar_no_legal_draft` stats.
- Grammar-aware speculation (#191, #201) is retired with it. If the depth
  controller is ever fixed, it would have to be rebuilt on upstream's engine.
- **Not runtime-validated.** This lands on `go build` plus a green unit suite.
  The MLX path needs a structured-output request against a built image before
  deploy; unit tests do not exercise MLX kernels.

## Amendment 2026-09-12: upstream drafts under a grammar again, and we keep it

The consequence above — "grammar-aware speculation is retired with it; if the depth controller is ever fixed, it
would have to be rebuilt on upstream's engine" — is what upstream then did. `4986e923` (v0.34.0) lets a
structured-output request draft: the grammar is enforced during verification, each draft position's logits are
masked before rejection sampling, and the drafts themselves stay unconstrained.

**Decision: align with upstream.** The v0.34.0 fold keeps upstream's behaviour as the default, and
`OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0` restores the previous gate for an operator who wants it. Nothing is offered to
`ollama/ollama`; this divergence is ours to carry.

**What the fold measured** (`tasks/upstream-sync-0.34.0.md`, four images over roughly thirty suite runs):

- **Speed.** 1.5 to 2.6 times the generation rate on structured output for the four larger models, which is the
  reason upstream made the change. Both runs shared the card with production and the MLX pins differed, so that
  ratio is context rather than a measurement.
- **Outputs.** Drafting flips knife-edge answers between runs. It does not move quality: without it the fold
  reproduces main's answers on every test, and the flips sit inside the run-to-run spread MLX already has
  ([ADR 0012](0012-benchmark-report-templates.md) §4, amended the same day).
- **Memory.** Drafting leaves MLX memory that no tracked array accounts for, and it grows across requests. This is
  not new in v0.34.0: main's own drafting does the same on think-on and format-less requests. Drafting under a
  grammar widens its reach to structured output, and the admission headroom cannot see any of it
  ([ADR 0034](0034-mlx-admission-prices-the-context-rung.md), amended the same day).

**Consequence.** The knob is the mitigation while the leak stands, not a fix. The fix belongs upstream in the
speculation path, and there is nothing to file there yet beyond a symptom.
