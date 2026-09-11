# TASK: fold upstream v0.34.0 into main

Branch `task/upstream-sync-0.34.0` (worktree `claude-scratch/wt-sync034`), cut from `main` at
`7d230fdc` (#293). Upstream v0.34.0 (`d8ab4b4f`, released 2026-09-05) adds 21 commits across
88 files. They are mostly `app/`, `cmd/`, `server/` and `x/mlxrunner`, where the changes are
structured output compiled as xgrammar structural tags and speculative decoding under a grammar.
First look: [upstream-sync-2026-09-04.md](upstream-sync-2026-09-04.md), section "Next fold".

**Not in this fold yet:** the MLX bump past ml-explore/mlx#4452, the fix for the idle runner that
pins a CPU core. It needs a native rebuild, and "don't rebuild yet" still stands.

## Status (2026-09-11, 23:15)

| gate | state |
|---|---|
| 1, merge | done, `ca3ff1db`, three conflicts resolved |
| 2, no-GPU tests | green after two fixes in `ba2eb4f1`; one upstream flake, two findings already on main |
| 3, image | Go-only swap `maxusai/ollama:sync-0.34.0-swap` (`0.33.3-dynres-26-gba2eb4f`), **valid for the GGUF half only** |
| 4, preflight `cuda-dynres-903` | running |
| 5, GGUF think-off against `ggmlmain_1_` | queued behind the preflight |
| 5, MLX think-off against `main276_` | **blocked: needs a native rebuild** |

## Conflicts and their resolution

- **`server/sched.go`, `runnerRef.LogValue`.** Kept the fork's leaf `logMu` (#289, #291) over
  upstream's `refMu.TryLock()`, which drops `name`, `inference`, `pid` and `num_ctx` at the eleven
  log sites that already hold `refMu`. Adopted upstream's `slices.Clone` of the GPU list inside
  the lock.
- **`x/mlxrunner/client.go`.** Took upstream's `requestGrammar`, which sends structured output as
  an xgrammar structural tag. This follows ADR 0033's adoption of upstream's grammar engine.
- **`x/mlxrunner/speculate.go`.** Took upstream's `accept()`, which lets a constrained session
  draft. Kept the fork's comments. Rewrote the `park()` note that said a constrained session never
  drafts; the parked step must still be masked, and it still is.

## Gate 2: what the tests caught

- **A silent merge regression, caught by the fork's own test.** Upstream moved `pid` into its
  `TryLock` block, which deleted it from the shared attribute list. That deletion merged without
  a conflict, so the fork's `LogValue` stopped logging `pid`.
  `TestRunnerRefLogValueKeepsFieldsUnderRefMu` (#291) failed on it. `pid` is restored in `ba2eb4f1`.
- **The fork's MLX format test needed a port, not a type fix.** Since 0.34.0, `requestGrammar`
  wraps every non-empty format in a structural tag, and `parseGrammar` accepts only such tags.
  `client_format_test.go` now asserts ADR 0009 on that path: empty formats stay unconstrained,
  every non-empty format survives as a constraint, and a malformed or unwrapped format is an
  error. Rejecting a format the compiler cannot honour, such as `"yaml"`, now happens in the
  native compile, and can only be checked live on a rebuilt payload.
- **Green after the fixes:** `go build ./...`; `go vet` on `x/mlxrunner/...` and `server`;
  `go test` on every package but one; `go test -race ./server/`; the preflight's
  `test_verdicts.py` (102 tests, 6 skipped). golangci-lint 2.13.2 reported only the type error
  that the port fixes.
- **Not ours: a flaky upstream test.** `cmd/launch`
  `TestCodexAppCountsOnlyOllamaRequestsInRegularProfile` fails about one run in three, on
  unmodified v0.34.0 as well, and upstream `main` has not touched it since. The cause:
  `codexAppRequestCursor.scanLocked` skips files whose mtime is earlier than the recorded start,
  and the test writes its session file right after recording `time.Now()`. The kernel stamps
  mtimes from a coarse clock. In this container, 1,999 of 2,000 freshly written files had an
  mtime earlier than the preceding `time.Now()`, by up to 1.5 ms. CI on this branch will hit it
  at the same rate.
- **Already on main, unchanged by the merge:** `go vet`'s unused `slices.Collect` result in
  `tokenizer/bytepairencoding_test.go`, and gofumpt diffs in two `integration/` test files. Those
  files carry the build tag `integration && release`, which keeps them outside CI's lint.

## Gate 3: why the Go-only swap covers only the GGUF half

The MLX, MLX-C and llama.cpp pins are byte-identical to v0.33.3. But upstream changed the xgrammar
shim that this repo builds itself: `x/mlxrunner/xgrammar/native`, which ships in the MLX payload
as `libollama_xgrammar.so`. The 0.34.0 loader requires five symbols the 0.33.3 library lacks,
all for speculative decoding under a grammar:

- `ollama_xgrammar_matcher_rollback` and `ollama_xgrammar_matcher_is_terminated`;
- `ollama_xgrammar_dynamic_matcher_new`, `_rollback` and `_is_terminated`.

On the swapped image the runner logs "Structured output is unavailable". It then answers every
MLX request that carries a format with HTTP 501. The vision suite sends formats, so an MLX
campaign on this image would measure the missing library, not the fold. The preflight and the
GGUF campaign run on llama.cpp's own grammar and are unaffected.

The gate's step 0 put the same four requests to gemma4:12b-nvfp4 on both images
(`preflight-runs/gate-sync034.log`, 2026-09-11 23:03 and 23:11):

| format | main, `main-a523d60b` | fold on the Go-only swap |
|---|---|---|
| `"json"` | 200, a JSON object | 501, structured output unavailable |
| a JSON Schema | 200, matching the schema | 501 |
| `"yaml"` | 400, "invalid format: expected "json" or a valid JSON Schema object" | 501 |
| `""` | 200, unconstrained text | 200, unconstrained text |

Main honours ADR 0009 live: `"yaml"` is refused before any generation. In 0.34.0 that check
moved into the native compile, because `requestGrammar` wraps `"yaml"` in a structural tag that
`parseGrammar` accepts. Whether the fold still answers `"yaml"` with an error, and with which
status, can only be seen on a rebuilt payload. It is the first MLX check after the rebuild.

The swap-validity check that passed this listed native paths by hand and omitted the shim. It now
names the shim and defers to the Dockerfile's `COPY` lines. The first-look section of the sync
doc carries the same correction.

## Decision: rebuild, with the MLX bump (Glenn, 2026-09-11 23:30)

The image is being rebuilt from `fbedf506` on the `bigdisk` builder (`claude-scratch/build-034.sh`,
tag `maxusai/ollama:sync-0.34.0`), with MLX pinned to `ce916dbb`: ml-explore/mlx#4452, which stops
the CUDA completion worker spinning a core after its first batch.

**The pin stops at that commit, not at MLX main.** The next CUDA-relevant commit, #4458, inserts a
`global_scale` parameter into `gather_qmm`'s C++ signature ahead of `sorted_indices`. MLX-C at our
pin (`c74db530`) passes `sorted_indices` and the stream positionally, so every MLX past #4458 needs
an MLX-C bump and a regenerated binding, and MLX-C has not adapted yet. The ten commits to
`ce916dbb` are the fix plus Metal, CPU and Python fixes and additive core APIs, so
`MLX_C_VERSION` is unchanged. Upstream ollama's main has moved neither pin.

**The build is long.** Every llama-server stage misses the cache, because the Dockerfile copies
all of `llama/compat` and its README changed since the `sync-0.33.3` build, and the `mlx` stage
rebuilds for the new pin and the new xgrammar shim.

**After the build, on the real image** (`claude-scratch/gate-sync034c.sh`):

1. The live format check, including what the fold answers to `"yaml"`.
2. The idle-CPU probe, against main's one core held at 100 %.
3. The five-model MLX think-off against `main276_`.
4. The preflight again, on the deploy candidate.
