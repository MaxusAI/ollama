# ADR 0013: Bound grammar repetition at llama.cpp's threshold, not at the context window

- **Status:** accepted, implemented + verified 2026-08-10 on fork `main`
  lineage. Amends the "scope of schema support" of
  [ADR 0009](0009-mlx-pure-go-constrained-sampling.md); 0009's converter
  semantics are otherwise unchanged.
- **Date:** 2026-08-10
- **Deciders:** MaxusAI fork maintainers

## Context

`x/structured` is a function-by-function port of llama.cpp b10091, but it
replaced **two** files, not one: the converter
(`common/json-schema-to-grammar.cpp` → `schema.go`) *and* the GBNF parser
(`src/llama-grammar.cpp` → `gbnf.go`). The repetition guards live in the
**parser**, so the port carried the converter's semantics and silently
dropped them:

- `src/llama-grammar.cpp:13` — `MAX_REPETITION_THRESHOLD 2000`
- `:652` — rejects `min_times`/`max_times` over the threshold
- `:494` — rejects a cumulative `n_prev_rules * total_rules` over it

`builder.repeat` therefore expanded `minItems`/`maxItems`/`minLength`/
`maxLength` literally, one grammar element per repetition. Measured at
`08d3784c`: `maxLength` 1e6 → 188 MB / 83 ms, strictly linear, so 3e8 →
tens of GB. One unauthenticated `POST /api/generate` with an ~80-byte body
reaches it through `Runner.Prepare` → `compileFormat`, which runs in the
runner's HTTP handler *before* the request is queued — so concurrent
requests stack rather than serialize. The GGUF twin rejected the identical
schema cleanly, which made this an engine divergence as well as a DoS.

Separately, a 121-byte schema whose `$ref` cycle consists only of tail
references compiled cleanly and then hung `NewMatcher`: `normalizeDepthLimit`
bounds expansion depth but not breadth.

## Decision

Reject, at compile time, any schema whose grammar cannot be materialized in
bounded work — the same "reject what it cannot constrain" principle ADR 0009
already applies to `pattern` and external `$ref`s.

- Both llama.cpp guards are ported behind `maxRepetitionThreshold = 2000`
  (`x/structured/gbnf.go`), including the **cumulative** product guard —
  without it, nested bounds (`maxItems:1999` of `maxLength:1999` strings)
  multiply out even though each bound alone looks reasonable. Upstream's
  error strings are reproduced verbatim.
- A grammar whose root normalizes to zero stable stacks is rejected with a
  named reason; expansion positions are memoized past a work budget so the
  breadth explosion cannot recur.

Both surface as ordinary `Compile` errors, which the runner already turns
into HTTP 400. Enforced by `TestGBNFRepetitionThreshold`,
`TestSchemaRepetitionBoundsRejected` (each with an allocation ceiling, so a
rejection that expands first still fails) and
`TestSchemaUnproductiveRefCycleRejected`.

## Options considered

- **Threshold 2000, matching llama.cpp** (chosen) — restores the property
  ADR 0009 exists to guarantee: both engines accept the same language. It is
  a *restored* guard, not new policy, so the fix is a port rather than an
  invention.
- **A bound derived from the model's `num_ctx`** (e.g. 256K less ~30%) —
  proposed by the maintainer in session, and rejected on three counts. It
  would make MLX *accept* schemas llama-server *rejects*, reintroducing the
  engine divergence in the opposite direction; it would not fix the DoS,
  since ~180K repetitions still cost ~35 MB per rule instance and nested
  repetitions multiply; and the units do not correspond — `maxLength` counts
  bytes in a byte-level grammar while `num_ctx` counts tokens, and the
  grammar is compiled before any image or token budget is resolved. The
  underlying intuition is sound (a bound beyond what context can hold is
  meaningless) but belongs in a friendlier validation applied to **both**
  engines, not in the MLX port alone.
- **Clamp silently to the threshold** — violates the fork's no-silent-drop
  principle: the caller asked for a constraint we would not be applying.
- **Rely on subprocess isolation** — an OOM kills the runner child, not the
  daemon, but it still takes the model offline for every in-flight request
  and costs a reload.

## Consequences

- A schema with a repetition bound over 2000, or whose nested bounds
  multiply past it, now returns 400 naming the reason on **both** engines.
  Previously: clean error on GGUF, runner OOM on MLX.
- Hostile schemas are rejected in microseconds having allocated ~0 MB
  (measured: 227 µs for `maxLength` 3e8; 361 µs / 1 MB for the nested case).
  Legitimate schemas — including productive `$ref` recursion — are unchanged.
- `maxRepetitionThreshold` is the single tuning knob, and moving it is a
  deliberate divergence from the pinned llama.cpp version.
- Follow-up: when the llama.cpp pin moves, this threshold tracks it, like the
  rest of the converter.
