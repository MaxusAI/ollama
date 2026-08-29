# TASK: merge upstream v0.33.2 into main

**Opened:** 2026-08-28. **Status:** assessed, not merged — blocked on one
architectural decision (below). Merge was attempted and aborted cleanly; `main`
is untouched.

## Scope

21 commits, `v0.33.0..v0.33.2`. Our sync point is `v0.33.0-dynres`.

**The payload pin moves: `LLAMA_CPP_VERSION` b10488 → b10630.** Unlike the
#217 sync, expectations are therefore *not* carried:

- **All seven compat patches apply cleanly at b10630** — verified with
  `git apply --check` against a fresh clone: 001, 002, 003, 004, 005, **801**,
  903. This was the feared cost and it is not there.
- `903` is **still required**: ggml-org/llama.cpp#27044 remains OPEN. Applying
  cleanly is not proof it is still *correct* at b10630 — the mmq code may have
  shifted under it. Needs a functional check, not just apply.
- **Preflight expectations need re-measurement.** `payload_pin` changes, so
  every recorded expectation is invalid until a build + full preflight run.

## What we gain

MLX, which is where the fork invests: Qwen3.8 Flash Next (`18ea9de0`),
structured output (`147509c0`), a Metal GPU timeout fix for slow storage
(`77e3b0ac`), exact type readers (`7623501f`), plus 142 upstream llama.cpp
builds and the Claude Desktop/proxy work (inert to the serving image).

**Nothing in the window touches the fp16-accumulate fault** — no qwen25vl,
fp16 or cuBLAS commits, and ollama/ollama#18070 is still open. The
`applyArchServerEnvs` gate stays load-bearing.

## Conflicts: 6 files, 12 hunks

| file | resolution |
|---|---|
| `.github/workflows/test.yaml` | union of both path filters — done |
| `llm/llama_server.go` | keep our `applyCompletionFormat` helper, adopt upstream's removal of the dead `req.Grammar` branch (`7027546c` deleted the field) — done |
| `llama/compat/README.md` | we added the 8xx band, upstream edited it in `ad94d529` — trivial union |
| `x/mlxrunner/runner.go` | h1 union imports; h3 **union** (our 181 lines of memory/cache config vs their 23-line `logitsWidth` — disjoint additions, not a collision); h4 keep our `fatalRunnerError`/`recoverRequest` plus their `defer request.Grammar.close()`; **h2 blocked** |
| `x/mlxrunner/pipeline.go` | **blocked** |
| `x/mlxrunner/speculate.go` | **blocked** |

Most of it is union of disjoint additions. The blocked hunks are all one thing.

## THE DECISION: whose constrained-sampling layer survives

Upstream has now implemented structured output in the MLX runner, and it
collides with the fork's:

| | fork (HEAD) | upstream v0.33.2 |
|---|---|---|
| request field | `Constraint *structured.Grammar` | `Grammar *grammarCompilation` |
| compile | `compileFormat()`, synchronous | `grammarEngine.prepare()`, async, overlaps tokenization/prefill |
| speculation | `constrainedStep` with matcher/vocab/pieces | `inner *pipelinedDecoder` + `grammar`; "a constrained session never drafts" |

Both reach the same *effective* behaviour — no drafting under constraint.
Ours is **correct but inert** (#201): it never proposes a draft because of a
cold-start deadlock in the depth controller, `drafted=0`, and the note records
three candidate fixes without picking one. Upstream's is simpler, maintained,
and async-compiles the grammar.

**Adopting upstream retires ADR 0009 (MLX pure-Go constrained sampling):**
~78 references across `x/`, and three test files —
`constrain_test.go`, `constrain_bench_test.go`, `client_format_test.go`.
It also needs ADR 0009's semantics re-established on upstream's engine: an
unsupported format must remain an **error**, never a silently dropped
constraint.

Options:

1. **Adopt upstream, retire ADR 0009.** Fewest fork deltas, gains async
   compile, loses the pure-Go path and the (unrealised) speculation upside.
   Needs a superseding ADR and rework of three test files.
2. **Keep ours, take upstream's non-grammar MLX changes only.** Preserves
   ADR 0009 and the three fixes still on the table, at the cost of carrying a
   growing delta against an area upstream is now actively developing.

Recommendation: **(1)**, because our layer is measured inert and upstream is
now maintaining this surface — but this retires a recorded decision and is
Glenn's call, not the merger's.

## Acceptance criteria

1. ☐ Decision above recorded, superseding ADR written if (1).
2. ☐ Merge with the resolutions in the table; zero remaining conflicts.
3. ☐ `go test ./server/ ./model/... ./llm/` green in `golang:1.26`
   (`-u 1000:1000`, `-buildvcs=false`).
4. ☐ `go test ./x/mlxrunner/` green — the semantic gate on the three
   mlxrunner conflicts.
5. ☐ 903 functionally revalidated at b10630, not just apply-clean.
6. ☐ Build, then **full preflight re-measurement** (the pin moved), and update
   `expectations.toml` `payload_pin`.
7. ☐ Metal half on the Apple host — deferred by Glenn, not a blocker here.
