# ADR 0026: A Qwen3.8 baseline records which effort directive it ran, not "default"

- **Status:** accepted 2026-08-16 on fork `main`; backported to
  `release/0.32.1-dynres` 2026-08-16 with the qwen3.8 renderer. Applies
  [ADR 0023](0023-think-mode-is-per-model-and-measured-on-policy.md)'s
  on-policy discipline to Qwen3.8's reasoning-effort axis; enforced by
  `TestQwen38ReasoningEffortMapping` (`model/renderers/qwen38_effort_test.go`).
- **Date:** 2026-08-16
- **Deciders:** MaxusAI fork maintainers

> **This backport supersedes a recorded decision.** `main`'s plan
> `docs/superpowers/plans/2026-08-16-qwen38-native-support.md` lists
> "Backporting to `release/0.32.1-dynres`" under **Out of scope**, reasoning that
> the lineage has no `qwen35` family arch row in its ROCm preflight profile, and
> asks for that to be recorded as a deliberate "no" under **ADR 0006**
> (`main`-only; named rather than linked, per the convention in
> [ADR 0011](0011-preflight-expectations-are-versioned-code.md)).
>
> **That reason is still true**, and this backport does not repair it — qwen3.8
> remains unmeasured on gfx1151 (see Lineage notes). What changed is the demand:
> the deployed host cannot serve qwen3.8 at all without the renderer, and the
> alternative — moving the lineage past 0.32.1 — is blocked by the AMD upgrade
> gate (`docs/maxusai/amd-upgrade-gate.md`, on `main`). The backport was taken as
> the gate-neutral option: fork Go code adapted to the pinned base, per ADR 0006's
> cherry-pick route, moving no payload.
>
> Scope of the backport: the qwen3.8 renderer and its tests, the parser changes
> it requires, a truncation-recovery fallback in `server/chatPrompt`, and
> qwen3.8 detection in the safetensors importer. It does **not** add a preflight
> baseline. `main`'s plan should be updated to record the reversal.

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
think to `true` for any thinking-capable model (`:467`, `:2945` on this
lineage), and Qwen3.8 declares `thinking` in its published capabilities. So
omitting `think` and sending `think:true` produce byte-identical prompts; the
xhigh branch fires only for direct Go callers.

This was initially recorded the other way round — that `think:true` ran at a
quieter "medium" while omitting `think` ran at the publisher's xhigh, making
the explicit request weaker than the implicit one. That is wrong, and it was
wrong in the direction that would have invalidated a benchmark. The empirical
matrix above is the record.

The publisher's own template defaults to xhigh
(`reasoning_effort|default('xhigh')`), which on this fork is reachable **only**
via `think:"high"` or `think:"max"` — the literal string `"xhigh"` is rejected
by `ThinkValue.IsValid()`.

The mapping reproduces identically here: `api.ThinkValue` is byte-identical
between `main` and this lineage, and `model/renderers/qwen35.go` was taken from
`main` unmodified.

## Decision

A Qwen3.8 measurement records the **directive it emitted**, from the table
above, not the request field it sent and not the word "default".

- The preflight harness sends `think=True` (`checks.py:375`), so its runs are
  **no directive** — the model's packaged behaviour, thinking on. That is a
  legitimate on-policy baseline under ADR 0023 and needs no change.
- A run wanting the publisher's default must send `think:"high"` and be
  recorded as **xhigh**, not as "default".
- `"medium"` is not a level. Do not record a run as medium-effort; it is the
  no-directive case under another name.

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
  effort levels. That matters more here than on `main`: qwen3.6 is one of the
  three families [ADR 0025](0025-think-stays-off-on-gfx1151.md) measured on
  this host. Cross-family effort comparisons need the column to be read.
- The mapping is pinned by a test, so an upstream change to
  `qwen38ReasoningInstructions` or to `ThinkValue.String()` breaks CI rather
  than silently re-labelling every future measurement.
- The unreachable `nil` branch is left as upstream wrote it. It is dead over
  HTTP but correct for Go callers, and removing it would be fork divergence
  for no runtime gain.

## What the backport changed in shared code

Three of the ported changes sit on paths this lineage already serves with
`gemma4`, `nemotron3` and `qwen3.6`. None was required by those models; each
arrived because qwen3.8 needs it. Recorded here because CI cannot catch them —
no test pinned the previous behaviour.

1. **`model/parsers/qwen35.go` — leading-whitespace trim.** `Qwen35Parser` is
   the `"qwen3.5"` parser, which qwen3.6/`qwen35moe` also use. Required for
   qwen3.8 (`TestQwen38ParserReferenceContinuationAtEverySplit/split_7` fails
   without it). Effect on qwen3.6: when the model emits a literal `<think>` that
   ends a streamed chunk, leading whitespace is now stripped from the `thinking`
   field — `"\nNeed weather."` becomes `"Need weather."`. Cosmetic, and it
   removes a chunk-boundary non-determinism.
2. **`model/parsers/qwen35.go` — doubled `<think>` no longer double-stripped.**
   The same hunk moves `allowLeadingThinkOpenTag = false` above the `after == ""`
   early return, so the strip can no longer run twice. Chunks
   `"<think>", "<think>", "hi"` previously yielded `thinking = "hi"` and now
   yield `thinking = "<think>hi"`. This aligns the code with its own documented
   contract ("Strip at most one such tag") but is a fidelity regression for a
   pathological emission. No served checkpoint is known to produce it. Left as
   upstream wrote it rather than diverging a shared parser for an unobserved case.
3. **`server/prompt.go` — truncation-recovery fallback.** `chatPrompt` no longer
   fails outright when a renderer rejects a truncation candidate; it falls back
   to the last window that rendered. Scoped to renderer-driven models
   (`m.Config.Renderer != ""`), so template-driven models still surface their
   errors rather than silently serving an untruncated conversation. The returned
   window may exceed `NumCtx` — that check is a heuristic and llama-server does
   the real truncation, so an over-long window degrades better than a dead
   request, which under ADR 0004's double request would die mid-stream.

`model/renderers/qwen35.go`'s `think != nil` → `think != nil && think.Value != nil`
also sits on a shared path (qwen3.5 and ornith), but is unreachable over HTTP:
`server/routes.go:467` and `:2945` coerce a nil think before rendering. It is a
correctness fix for direct Go callers only.

## Lineage notes

- **No `CARD_THINKING` clause.** `main`'s version rules out adding a
  `CARD_THINKING` entry for the `qwen3.8` family. That machinery lives in
  `docs/maxusai/vision-suite/sampling.py`, which does not exist on this
  lineage, so the clause has nothing to bind to here.
- **Qwen3.8 is unmeasured on gfx1151.** `expectations.toml`'s
  `[profiles.rocm-0-32-1-dynres]` lists `arches = ["nemotron_h_omni", "gemma4"]`
  and carries no `qwen35`-family row. Under ADR 0011 rule 4 an unmeasured
  `(platform, arch)` pair reports `NEEDS_BASELINE` and exits 4. Adding an
  `[expect.…]` block is not enough on its own — the profile's `arches` list must
  be extended in the same change. Do **not** carry `main`'s Apple-Silicon
  `qwen35` numbers across: rule 4 calls that "the failure this file exists to
  prevent", and note it is a rule for humans, not a gate — `test_verdicts.py`
  checks ladder arithmetic and would not catch a transplanted baseline.
- **Expected to load here; not yet loaded here.** The payload has the
  architecture — b9888 registers `LLM_ARCH_QWEN35`/`LLM_ARCH_QWEN35MOE`, the
  published qwen3.8 GGUF declares `model_family: "qwen35"`, `llm/llama_server.go`
  (`:823`, `:839`) knows the family, and `llama/compat` handles its metadata
  (`README.md:98`). None of that is a qwen3.8 load on this host, and `main`'s
  plan records the b9888 payload for this family as **unverified**. Treat
  servability as expected, not established, until a run exists.
- **ADR 0025 tension, not conflict.** `newQwen38Renderer` sets `isThinking: true`,
  while ADR 0025 keeps `think` off on gfx1151. No conflict: 0025's decision is
  scoped to the three *measured* families and states explicitly that any new
  family needs its own measurement. The renderer field is only a default for a
  nil think, which HTTP never delivers. An operator sending `think:false` gets
  thinking off and no directive.
