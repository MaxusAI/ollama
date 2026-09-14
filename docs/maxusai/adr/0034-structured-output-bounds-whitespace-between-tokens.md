# ADR 0034: Structured output bounds the whitespace between tokens

- **Status:** proposed 2026-09-14 (PR #301). Extends the structural-tag path
  adopted in [ADR 0033](0033-mlx-constrained-sampling-adopts-upstreams-engine.md);
  the same class of bound as [ADR 0013](0013-grammar-repetition-bounded-at-llama-cpp-parity.md),
  on a different axis.
- **Date:** 2026-09-14
- **Deciders:** MaxusAI fork maintainers

## Context

`requestGrammar` (`x/mlxrunner/client.go`) wraps a request's JSON schema into an
xgrammar structural tag. xgrammar reads `max_whitespace_cnt` from that tag's
`json_schema` format object (`cpp/structural_tag.cc`) and defaults it to null;
with no bound, `IndentManager::StartSeparator` compiles **every** separator to
`[ \n\t]*` (`cpp/json_schema_converter.cc`).

So the grammar permits an unlimited whitespace run between any two JSON tokens.
A model that stalls mid-object is not corrected by the constraint — it is
*licensed* by it, and emits newlines and indentation until `num_predict` or the
request timeout is exhausted. The answer is never closed, so the caller receives
`done_reason=length` and nothing it can parse.

**Measured 2026-09-14**, `gemma4:31b-nvfp4` on this runner, a 4,528-character
document under a 15-field schema: the decode filled its whole 1500-token budget,
**82% of the output whitespace**, the tail `"\n  "` repeated to the end, one
object started and never closed. Across a 174-document run, six documents were
lost this way — four to the 8192-token cap (~9.5 min each) and two to a
900-second timeout, one of them twice under retry. Other documents of the same
template and comparable size decoded in under a minute in the same run.

Two things make this expensive to diagnose from outside. It presents as a flaky
model rather than as a grammar that allows the stall; and at `temperature 0` a
retry is not an independent draw, so a document that stalls tends to keep
stalling while its neighbours are fine.

## Decision

`requestGrammar` sets `max_whitespace_cnt` on the `json_schema` format. The value
is **32**: a newline plus ten levels of two-space indent is 21 characters, so 32
is above any indentation a model would legitimately emit between two tokens, and
far below a stall, which runs to thousands. The bound applies per separator, not
per document, and does not touch the schema it wraps.

## Options considered

- **Bound the whitespace in the tag (chosen).** One field, in the place xgrammar
  already reads it, and it ends the stall at the grammar rather than at a
  timeout.
- **Leave it and rely on `num_predict` / the request timeout.** That is the
  current behaviour: the caller pays minutes per stalled request and gets an
  unclosed answer with no indication of why.
- **`any_whitespace=false` (compact output only).** Also ends the stall, but
  changes what the model is allowed to emit for every existing caller, and
  breaks any consumer that expects pretty-printed JSON.
- **Make the bound an option on the API.** A knob nobody sets by default; the
  stall would keep reaching users who never heard of it. A constant that can be
  promoted to a knob later is the smaller first step.

## Consequences

- A request whose schema genuinely needs more than 32 whitespace characters
  between two tokens would now be constrained. No such schema is known; if one
  appears, this becomes a knob.
- The bound is asserted by unit tests on the wire tag
  (`TestStructuralTagBoundsTheWhitespaceBetweenTokens`), not by a live decode.
  **Not yet verified end to end:** a build of this branch re-running the six
  documents that were lost is the confirmation, and it has not been done.
- Nothing changes for the GGUF path, which passes the schema to llama-server and
  is bounded by llama.cpp's own converter.
