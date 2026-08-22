# TASK: merge upstream/main (6bba484f, v0.32.15+3) into main

**Opened:** 2026-08-21. **Status:** OPEN — assessment complete, merge not
started. This document is the full handoff; the agent picking this up should
do the merge **on this branch** so the PR closes with the work.

## Scope

Merge `upstream/main` at `6bba484f` (v0.32.15-3-g6bba484f, fetched
2026-08-21) into `main`. Merge-base is `d67ad834` — our fork was fully
synced before this batch, so exactly **10 upstream commits** are in scope,
all authored 2026-08-18 → 2026-08-20. A dry-run `git merge-tree --write-tree
main upstream/main` (run 2026-08-21) shows **8 of 10 auto-merge; exactly two
files conflict.** Details below are from that dry run, not speculation.

## The 10 commits

| Hash | Subject | Assessment |
|---|---|---|
| `e0c95a5f` | server: don't wedge chat and generate on a mid-stream parser error (#17883) | **Headline item — see below** |
| `a5165c53` | Add a model metadata cache to reduce per-request overhead (#17752) | High value; benchmark-relevant behavior change |
| `0bb09259` | mlx update (#17850) | MLX pin bump + mlx-c 0.32.1 regen compat patch |
| `e92b7855` | mlx update (#17886) | second MLX pin bump |
| `cd370440` | llama.cpp update (#17851) | b10434 → b10488 |
| `4e134213` | mlx: fix mac assumptions on linux/windows (#17898) | We already fixed the RPATH half in-fork; their Windows `LoadLibraryExA` half is new |
| `b8a62724` | qwen3.8: normalize system messages (#17855) | Low risk — see qwen3.8 note |
| `d1bd15cc` | ci: plumb temporary MLX patch through to docker stages (#17874) | 1-line Dockerfile; auto-merges past our +21 fork lines |
| `b7871fc0` | app: add desktop onboarding flow (#17853) | app/ only, irrelevant to server work |
| `6bba484f` | lint fixes (#17897) | trivial |

MLX pinned commit: `adf21dea…` → `27fec909…` (two bumps, plus
`mlx/compat/0001-mlx-c-regen-0.32.1.patch`). llama.cpp: `b10434` → `b10488`.

## Why this merge is worth doing

**`e0c95a5f` is the reason to do this promptly.** When a builtin parser
rejects model output mid-stream, the completion callback writes the error to
an unbuffered channel and returns; the callback can't stop generation, so
the next chunk re-enters, hits the same error, and blocks on a channel the
consumer stopped reading after its 500. The goroutine leaks and **the runner
request is never released** — retries of the same prompt hang with no log
output. Only thinking mode wedged (final-chunk parse failures were already
terminal). This is precisely our exposure: the fork carries its own parser
changes (`model/parsers/nemotron3nano.go`, `model/parsers/qwen35.go`) and
runs hours-long thinking-mode benchmark campaigns where one wedged runner
poisons a whole run. Upstream's fix: record the parse error, cancel the
completion, report once the completion returns. Ships with
`server/routes_parse_error_test.go` (140 lines).

**`a5165c53`** caches fully resolved model metadata keyed by manifest digest
(+ GoTemplate env), with singleflight — cuts per-request manifest/GGUF
parsing. This lowers the per-request latency floor under our req/h numbers:
a provenance-relevant behavior change, already covered by SPEC H11's
per-cell `server_version`.

**MLX + llama.cpp bumps** are routine but numerically risky for the gemma4
MLX vision path — hence the golden-test gate below.

## Conflict 1 — `x/mlxrunner/mlx/CMakeLists.txt` (trivial)

Both sides fixed the same bug: the Mach-O `@loader_path` RPATH spelling
written on ELF. Ours is `if(APPLE) … elseif(UNIX)` with a comment explaining
the failure (references `cmake/mlx/CMakeLists.txt`); upstream's new fix is
`if(APPLE) … else()`. Both land on `$ORIGIN` for non-Apple.
**Resolution: keep ours** (comment + stricter guard). Do keep upstream's
sibling change in `x/mlxrunner/mlx/dynamic.c` (Windows
`SetDllDirectoryA` + `LoadLibraryExA` DLL-search fix) — it auto-merges and
does not overlap our work.

## Conflict 2 — `server/routes.go` (the real work, but contained)

Two conflict regions, **both in `GenerateHandler`** — exactly where the
fork's structured-outputs marker-flow state machine lives (pass-1/pass-2,
`structuredOutputsState_*`, transition-metrics refinement per ADR 0010).
The ChatHandler half of upstream's fix auto-merges cleanly.

What upstream changed in that callback:

1. `ctx, cancel := context.WithCancel(c.Request.Context())` + `defer
   cancel()` before the completion starts (GenerateHandler previously had no
   cancel func at all), and a `var parserErr error`.
2. On builtin-parser `Add()` error: `parserErr = err; cancel(); return`
   instead of `ch <- gin.H{"error": …}; return`.
3. After the completion returns: OOM/error reporting is gated on
   `parserErr == nil`, then `if parserErr != nil { ch <- gin.H{"error":
   parserErr.Error()} }` — the error is reported exactly once, after the
   runner request is released.

Resolution: thread that pattern through our extended callback. Every place
our marker-flow code path can get a parser error mid-stream must set
`parserErr` + `cancel()` rather than writing to `ch` from inside the
callback. Watch the pass-one → pass-two transition: a parse error during
pass one must not leave the pass-two restart armed. Our fork's
`s.sched.expireRunnersForRuntimeOOM` call sits in the second conflict hunk —
keep it, gated on `parserErr == nil` as upstream does.

## qwen3.8 note (`b8a62724`)

`normalizeQwen38Messages` now folds **all** system/developer messages
anywhere in the history into one leading system turn (previously: only a
leading run, and only when a developer message was present). Requests with
zero or one leading system message are explicitly returned unchanged — so
single-system-prompt benchmark probes are unaffected; only multi-system /
developer-role conversations render differently. The fork adds
`model/renderers/qwen38_effort_test.go` against this same file — it must be
re-run (and possibly updated) after the merge.

## Acceptance criteria (in order)

1. `git merge upstream/main` on this branch; only the two conflicts above;
   resolutions as specified.
2. `go test ./server/ ./model/renderers/ ./model/parsers/` — green,
   including upstream's new `routes_parse_error_test.go` and our
   `routes_generate_test.go`, `sched_headofline_test.go`,
   `qwen38_effort_test.go`. (Known baseline: `go build ./...` fails on the
   app/dist embed — pre-existing, not this merge's problem; test the listed
   packages, not `./...`.)
3. MLX vision golden tests (`x/mlxrunner/vision_golden_test.go`, goldens for
   12b/26b/31b) — the MLX pin bump is the numeric risk. Respect the
   single-owner-thread rule for MLX tests.
4. Build an image and run the pre-deploy preflight harness
   (`docs/maxusai/vision-suite/preflight/`) against a benchmark host before
   any deploy; both llama.cpp (GGUF q4_K_M/q8 tags) and MLX paths changed.
5. Note in the merge commit body that `server_version` provenance (SPEC H11)
   is the comparability boundary for any benchmark cells measured on the new
   build (metadata cache + renderer change).
