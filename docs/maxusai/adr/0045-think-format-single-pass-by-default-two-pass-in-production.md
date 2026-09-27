# ADR 0045: think+format follows upstream's single pass by default; production runs two-pass, because on MLX that is what lets the thinking draft

- **Status:** accepted 2026-09-28 for the two decisions it records. Both are the maintainer's.
  - The fold keeps upstream's single pass as the default, with ADR 0004's flow behind `OLLAMA_FORMAT_TWO_PASS=1`
    ([#375](https://github.com/MaxusAI/ollama/pull/375)).
  - Production runs that switch. Decided 2026-09-27; deployed on CUDA and gfx1151 on 2026-09-28.

  The retirement condition at the end is proposed.
- **Supersedes:** [ADR 0004](0004-routes-layer-think-format-double-request.md) as the statement of which flow the fork
  runs. ADR 0004, with [ADR 0002](0002-deferred-format-constraining.md) and
  [ADR 0010](0010-transition-flow-metrics-reconstruction.md), remains the reference for how the two-pass flow works.
- **Related:** [ADR 0033](0033-mlx-constrained-sampling-adopts-upstreams-engine.md). Production keeps its knob at 0.
- **Date:** 2026-09-28
- **Deciders:** the maintainer. Measured on the three hosts in the v0.34.4 fold:
  [#375](https://github.com/MaxusAI/ollama/pull/375) and the [fold record](../tasks/upstream-sync-0.34.4.md).

## Context

Since ADR 0004, a request with both `think` and `format` has run in two passes at the routes layer:

- **Pass one** thinks with no grammar. It stops at the model's think-close marker, or at the thinking→content
  transition when the model has no marker.
- **Pass two** answers under the grammar.

Upstream v0.34.4 replaced its own two-pass transition flow with one pass on both engines (`5a0ff3116`, "server:
apply structured outputs in a single pass on thinking models"):

- The parsers report the strings that close their thinking (`ThinkingClose`), and the server names them on one
  completion request.
- llama-server wraps its schema grammar in a GBNF rule that allows free text first.
- MLX sends XGrammar a structural tag: free text (`any_text`, excluding the closings), then an optional closing
  followed by the schema.

Upstream's case for it:
- one prefill instead of two;
- no boundary chunk dropped at the transition;
- no stray first token in MLX's JSON;
- all 18 parsers covered, where the fork's marker hook covered two.

The fold took the single pass as the default, and kept ADR 0004's flow behind `OLLAMA_FORMAT_TWO_PASS=1` as a
rollback in case the single pass regressed on a served model. The fold record's think+format section shows how the
two were merged. Which flow production should run was left open, as the record's open item 3.

One interaction decides it. On MLX, `draftingEnabled` (`mlxrunner/speculate.go`) is set once per request, to
`request.Grammar == nil || draftUnderGrammar`.
- **Production runs `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`**, the knob that ADR 0033's 2026-09-12 amendment added.
  Drafting under a grammar retains memory on the qwen3.5 family's think-off structured image requests: in the 0.34.1
  fold's `held` series, qwen3.6 grew 0.60 GiB per request.
- **A single-pass request carries its grammar from the first token.** So under that knob it never drafts, even while
  it thinks.
- **Two-pass's first pass carries no grammar.** It drafts the thinking, and only the short answer runs undrafted.

## What was measured

All of this ran in the v0.34.4 fold, on the fold's own images. The fold record and #375 carry the generator output.

- **Speed, on CUDA: the drafting probe.** Think+format at temperature 0, with every request thinking up to an
  8,192-token cap, on production's environment. There were three arms: F0 (single pass, knob 0), P0 (two-pass,
  knob 0) and F1 (single pass, knob 1). P0 thinks faster than F0:

  | model | condition | P0 | F0 | P0 over F0 |
  |---|---|---|---|---|
  | qwen3.8 | quiet | 44.1 tok/s | 26.4 tok/s | 1.7× |
  | gemma4:31b | uncontended | 48.0 tok/s | 32.7–33.0 tok/s | 1.46× |
  | gemma4:31b | same heavy peer | 40.5 tok/s | 24.5 tok/s | 1.65× |

  F1 and P0 are level within the spread. So the flow changes speed only through the knob.
- **Loops on MLX-Metal: the aligned think-on protocol.** Greedy, the full ladder, and a cold server at every rung.
  Cases left unfinished at 131072, out of 27 (Metal's comment on
  [#375](https://github.com/MaxusAI/ollama/pull/375#issuecomment-5856289579)):

  | model | single pass, never drafts | two-pass | single pass, knob 1 |
  |---|---|---|---|
  | gemma4:26b | 4 | 1 | 1 |
  | gemma4:31b | 2 | 0 | 0 |
  | qwen3.6 (cannot draft on MLX) | 6 | 7 | — |
  | qwen3.8 | 0 | 0 | 0 |

- **Loops on MLX-CUDA: none from the flow.** The single pass loops no more than two-pass, whether the suite runs cases
  in order or each case alone on a cold server. Every arm left 1 of 13 cases unfinished; one two-pass repeat left 2.
  - A cold load's own trajectory varies more than the flows do. The same greedy request finished between 1 and 3
    times in 5 cold draws, depending on the prompt (open item 8).
- **Loops on GGUF: none from the flow (gfx1151).** Neither flow drafts. On all five models the two flows leave the
  same cases unfinished, and those loops come from the prompt.
  - Quality has no net direction. qwen3.6 leans toward two-pass: 3 of 4 moved cases under `q8_0`, and 5 of 7 under
    f16 with one mixed. nemotron3 leans toward the single pass on the bbox contract (`contract_followed` 23/40
    against 15/40, p ≈ 0.1, not established). The other three models do not move.
  - The KV type moves more cases than the flow does.

So on MLX the question is whether the thinking drafts, not which flow runs. On GGUF, the two flows are equal.

## Decision

1. **The code default stays upstream's single pass.** The fork does not diverge from upstream's flow without a
   measured reason on a served surface. Where nothing drafts, there is none against the single pass.
   `OLLAMA_FORMAT_TWO_PASS=1` selects ADR 0004's flow.
2. **Production runs two-pass.**
   - The CUDA and gfx1151 containers set `OLLAMA_FORMAT_TWO_PASS=1`. They were deployed on 2026-09-28 at 07:37 and
     07:39 ([#390](https://github.com/MaxusAI/ollama/pull/390), [#391](https://github.com/MaxusAI/ollama/pull/391)).
   - On MLX this restores the thinking speed production had before v0.34.4, and Metal's lower gemma4 loop counts. It
     also keeps `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0` and the protection that knob gives.
   - gfx1151 serves GGUF, where the flows tie, and runs two-pass for the same behaviour as CUDA.
   - The Metal host's deploy is decided separately.
3. **Every deploy script sets the switch explicitly, and checks it.** A deploy that mirrors the live container by
   `docker inspect` would otherwise run whatever flow the code defaults to. The CUDA script adds the variable, as it
   adds `OLLAMA_KV_CACHE_TYPE=f16` ([#387](https://github.com/MaxusAI/ollama/pull/387)). It refuses a live container
   that sets another value, and rolls back unless the new server's startup config reads
   `OLLAMA_FORMAT_TWO_PASS:true`. gfx1151's deploy script does the same for both variables, and rolls back unless the
   config reads both. That host's compose file carries them too.

### What each flow does with the budget

This table records the budget semantics that open item 3 asked for.

| | single pass (the default) | two-pass (`OLLAMA_FORMAT_TWO_PASS=1`, production) |
|---|---|---|
| `num_predict` | bounds the whole output | bounds each pass, so the total can exceed it (8,290 against 8,192 on qwen3.6 `bbox_contract_reasoning`) |
| format, tools and thinking together (chat) | a tool call cannot replace the formatted answer (harmony excepted) | a tool call after the thinking can, because chat keeps the transition flow when tools are present |
| raw generate | never defers: a raw prompt names no closing strings, so the format applies from token 0 | can defer at a think-close marker (ADR 0004) |
| an EOS inside the thinking | returns only the thinking: `response:""`, `done_reason:"stop"` | continues to pass two (ADR 0004, decision 4) |
| MLX drafting, knob at 0 | none, for the whole request | pass one drafts; the answer does not |
| prefills | one | two: pass two re-submits pass one's tokens |
| metrics | the runner's own | pass one's prompt count, and eval counts and durations summed over both passes (ADR 0004, 0010) |

## Consequences

- **Production thinks at P0's speed.** It gets Metal's lower gemma4 loop counts, and the knob stays at 0 on every
  host.
- **ADR 0004's machinery is now production's flow**, not a superset of upstream's.
  - The fold restored the lower-layer hooks that upstream deleted: `IncludeIntermediateMetrics`, llama-server's
    per-token timings and per-chunk metrics, and the MLX request fields. They stay inert unless the switch is on.
  - The nine two-pass route tests pin the switch on, and the single-pass tests pin it off.
- **A fork build without the switch runs upstream's single pass.**
  - With the knob at its default, it drafts under the grammar. That is F1: fast, but with the retention on the
    qwen3.5 family.
  - With the knob at 0, it does not draft. That is F0: slow.
- **The next fold's think-on comparisons run both flows**, because the code default and production now differ.

## Retirement (proposed)

The switch, and with it ADR 0004's flow, retires when a single pass drafts its thinking without the retention. The
likely route is an `mlxrunner` change: draft while the grammar is still in its free-thinking element, and stop at the
closing. That is what pass one does today, done in one pass. Its gates:

- the size ladder (`leak-repro5.sh`) holds flat on the qwen3.5 family;
- the drafting probe shows the single pass matching P0;
- the think-on protocol on MLX shows the single pass matching two-pass's loop counts, on Metal, where the difference
  showed.

When those pass, production drops the switch in one deploy. The retirement register's row for ADR 0002, 0004 and
0010 then moves to "retire now".
