# ADR 0026: A Qwen3.8 baseline records which effort directive it ran, not "default"

- **Status:** accepted 2026-08-16, on fork `main`. Applies
  [ADR 0023](0023-think-mode-is-per-model-and-measured-on-policy.md)'s
  on-policy discipline to Qwen3.8's reasoning-effort axis; enforced by
  `TestQwen38ReasoningEffortMapping` (`model/renderers/qwen38_effort_test.go`).
- **Date:** 2026-08-16
- **Deciders:** MaxusAI fork maintainers

## Context

Qwen3.8's chat template adds a reasoning-effort axis. Upstream folds it onto
the existing `api.ThinkValue` rather than adding an API field, and the mapping
is not what it looks like. Measured by rendering, not by reading:

| `think` | directive emitted |
| --- | --- |
| `nil` | **xhigh** |
| `true` | **none** |
| `"medium"` | **none** |
| `"low"` | low |
| `"high"` / `"max"` | **xhigh** |
| `false` | none — and thinking is off entirely |

Two traps sit in that table.

`ThinkValue.String()` reports **`"medium"` for the boolean `true`**
(`api/types.go`), and the renderer treats `"medium"` as *no directive at all*
rather than a middle setting. So `think:"medium"` and `think:true` are the
same prompt, and neither is a "medium effort" run.

The `nil` row is **unreachable over HTTP**. `server/routes.go` coerces a nil
think to `true` for any thinking-capable model (`:462`, `:2927`), and Qwen3.8
declares `thinking` in its published capabilities. So omitting `think` and
sending `think:true` produce byte-identical prompts; the xhigh branch fires
only for direct Go callers.

This was initially recorded the other way round — that `think:true` ran at a
quieter "medium" while omitting `think` ran at the publisher's xhigh, making
the explicit request weaker than the implicit one. That is wrong, and it was
wrong in the direction that would have invalidated a benchmark. The empirical
matrix above is the record.

The publisher's own template defaults to xhigh
(`reasoning_effort|default('xhigh')`), which on this fork is reachable **only**
via `think:"high"` or `think:"max"` — the literal string `"xhigh"` is rejected
by `ThinkValue.IsValid()`.

## Decision

A Qwen3.8 measurement records the **directive it emitted**, from the table
above, not the request field it sent and not the word "default".

- The preflight harness sends `think=True` (`checks.py`), so its runs are
  **no directive** — the model's packaged behaviour, thinking on. That is a
  legitimate on-policy baseline under ADR 0023 and needs no change.
- A run wanting the publisher's default must send `think:"high"` and be
  recorded as **xhigh**, not as "default".
- `"medium"` is not a level. Do not record a run as medium-effort; it is the
  no-directive case under another name.
- Do **not** add a `CARD_THINKING` entry for the `qwen3.8` family by analogy
  to qwen3.6. `family()` falling through to packaged defaults is the correct
  behaviour here and is explicitly admissible under ADR 0023, because Qwen3.8
  ships its own effort handling.

## Options considered

- **Record the emitted directive** (chosen) — the only label that survives the
  `true`/`"medium"`/omitted collapse, and the only one that distinguishes a
  packaged-default run from an xhigh run. Costs one column in the record.
- **Record the `think` request value** — what a naive harness would log. It
  reports three different-looking values (absent, `true`, `"medium"`) for one
  identical prompt, and calls the weakest setting "medium". Rejected: it
  encodes the trap instead of the fact.
- **Force `think:"high"` for all Qwen3.8 baselines**, to match the publisher —
  makes fork numbers incomparable with ordinary fork traffic, which never
  sends it. Rejected; the useful baseline is what users actually get.
- **Add a fork-local default so the API default becomes xhigh** — would mean
  diverging in `model/renderers/qwen35.go`, a file the fork has deliberately
  kept identical to upstream, to fight a mapping upstream owns. Rejected.

## Consequences

- Baselines gain an explicit effort column; a record saying only "default" is
  incomplete and should be re-labelled or re-measured before it is cited.
- Comparing a Qwen3.8 cell against a qwen3.6 cell compares
  no-directive-with-thinking-on against qwen3.6's think verdict, not two equal
  effort levels. Cross-family effort comparisons need the column to be read.
- The mapping is pinned by a test, so an upstream change to
  `qwen38ReasoningInstructions` or to `ThinkValue.String()` breaks CI rather
  than silently re-labelling every future measurement.
- The unreachable `nil` branch is left as upstream wrote it. It is dead over
  HTTP but correct for Go callers, and removing it would be fork divergence
  for no runtime gain.
