# ADR 0035: Structured output bounds the whitespace between tokens

- **Status:** proposed 2026-09-14 (PR #301). Extends the structural-tag path
  adopted in [ADR 0033](0033-mlx-constrained-sampling-adopts-upstreams-engine.md);
  the same class of bound as [ADR 0013](0013-grammar-repetition-bounded-at-llama-cpp-parity.md),
  on a different axis.
- **Date:** 2026-09-14
- **Deciders:** MaxusAI fork maintainers

## Context

`requestGrammar` (`mlxrunner/client.go`) wraps a request's JSON schema into an
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

## Amendment 2026-10-12: 22, the GGUF run, and both bounds checked by a matcher

**What moved it.** muse-glimmer on MLX in single pass (`OLLAMA_FORMAT_TWO_PASS=0`), answering after its message header, padded the first and last slot of every object to the bound. `format_schema` on `30b-nvfp4` returned `{` + 33 spaces, the value, about 23 spaces and `}` (#463, #464). The bound held: stock 0.40.2 has no bound and filled all 16384 tokens with whitespace (FAIL). The cost of the padding was the bound itself.

**Decision, amended:**
- **MLX:** `max_whitespace_cnt` is **22**. That is the longest run llama.cpp allows for a JSON schema: `SPACE_RULE`, two newlines and 20 spaces or tabs. Both runners now bound whitespace alike. A newline plus ten levels of two-space indent (21 characters) still fits.
- **GGUF, `format: "json"`:** the GGUF path is not wholly bounded. A schema goes to llama-server's converter, which is bounded. But `format: "json"` uses the fork's own grammar (`grammarJSON`, `llm/llama_server.go`), and its `ws ::= ([ \t\n] ws)?` was unbounded. It is now `ws ::= (" " | "\n"{1,2} [ \t]{0,20})?`: llama.cpp's rule, written as an optional group, because xgrammar's parser rejects a leading empty alternative. Placement is unchanged: whitespace still follows literal characters only.
- **Not taken: patching xgrammar to llama.cpp's exact rule.** On MLX that would cut glimmer's padding to one space per slot. It costs a native patch on every xgrammar bump, and it moves every MLX format request's mask. xgrammar v0.2.7 hardcodes `any_whitespace=true` for structural-tag `json_schema` elements (`cpp/structural_tag.cc`), so only the count can be set. Revisit if xgrammar adds a whitespace-rule option (mlc-ai/xgrammar#837 is open).

**Verification, amended:**
- `TestRequestGrammarWhitespaceRunMatchesLlamaCpp` drives `requestGrammar`'s tag through xgrammar one byte at a time, for a plain format, a format after thinking, and `"json"` after a header. It checks that 22 spaces are accepted, 23 refused, and two newlines plus 20 spaces accepted.
- `TestGrammarJSONBoundsWhitespace` does the same for `grammarJSON`, through xgrammar's grammar parser. llama.cpp's engine has no Go entry point.
- Both fail on the old grammars. They run wherever the MLX payload is built and skip elsewhere. The "not yet verified end to end" above still holds for MLX documents; the real-model check of this amendment is the GGUF and MLX A/B on the branch's PR.

**Upstream:**
- ollama/ollama has no bound on MLX. ollama/ollama#18567 (a whitespace runaway) was closed as standard behaviour.
- ollama/ollama#18569 (a cap of 32) was closed untested.
- The fork's bound has not been offered upstream.
