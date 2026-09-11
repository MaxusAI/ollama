# TASK: fold upstream v0.34.0 into main

Branch `task/upstream-sync-0.34.0` (worktree `claude-scratch/wt-sync034`), cut from `main` at
`7d230fdc` (#293). Upstream v0.34.0 (`d8ab4b4f`, released 2026-09-05) adds 21 commits across
88 files. They are mostly `app/`, `cmd/`, `server/` and `x/mlxrunner`, where the changes are
structured output compiled as xgrammar structural tags and speculative decoding under a grammar.
First look: [upstream-sync-2026-09-04.md](upstream-sync-2026-09-04.md), section "Next fold".

**Not in this fold yet:** the MLX bump past ml-explore/mlx#4452, the fix for the idle runner that
pins a CPU core. It needs a native rebuild, and "don't rebuild yet" still stands.

## Status (2026-09-12, 01:35)

| gate | state |
|---|---|
| 1, merge | done, `ca3ff1db`, three conflicts resolved |
| 2, no-GPU tests | green after two fixes in `ba2eb4f1`; one upstream flake, two findings already on main |
| 3, image | Go-only swap `maxusai/ollama:sync-0.34.0-swap` (`0.33.3-dynres-26-gba2eb4f`), **valid for the GGUF half only** |
| 4, preflight `cuda-dynres-903` | PASS 20, SKIP 8 on the swap image, the same as main; repeated on the rebuilt image |
| 5, GGUF think-off against `ggmlmain_1_` | **green**: 8 suites, no OOM, no error; every quality row identical to main |
| MLX format check and idle-CPU probe | **passed** on the rebuilt MLX payload (below) |
| 5, MLX think-off against `main276_` | running on the payload-swap image since 01:33 |

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

## Gates 4 and 5, GGUF half: identical to main on quality

The GGUF (llama-server) path runs on llama.cpp's own grammar, so the swap image measures it
validly. Both gates ran on it on 2026-09-11 (`claude-scratch/gate-sync034b.sh`):

- **Preflight** `cuda-dynres-903`, 23:16 to 23:31: passed 20 checks and skipped 8, the same
  counts as main's run on 2026-09-06 (`preflight/runs/full-sync034.json`).
- **GGUF think-off campaign**, eight models, 23:31 to 00:41
  (`preflight-runs/ggml034_1_thinkfalse.log`): 8 suites, no OOM, no error, every cell converged at
  8192.

It was rendered with the ADR 0012 generators (`claude-scratch/render-ggml034.sh`, output in
`preflight-runs/ggml034-render-thinkfalse.md`), comparing main's `ggmlmain_1_1_` with the fold's
`ggml034_1_1_`:

- Every quality row of the eight head-to-head tables matches main: 56 rows, no differences.
- The contract matrices are identical.
- The T1 quality columns match, answer-token counts included. Only throughput and latency
  differ. The fold's campaign shared the host with a teacher leg on GPU 0 and, from 23:45, with
  the image build, at a load average of about 100 on 32 cores. Those columns measure the host,
  not the fold.

## MLX half on the rebuilt MLX payload (2026-09-12)

The full image crawls at the lowest CPU weight beside a teacher leg, so the MLX gates run on
`maxusai/ollama:sync-0.34.0-mlxswap` instead (`claude-scratch/gate-sync034d.sh`, log
`preflight-runs/gate-sync034d.log`). It is main's image with two things swapped in:

- the fold's Go binary, `0.33.3-dynres-30-g5404bec`;
- the fold's freshly built MLX payload, exported from the full build's own `mlx` stage through the
  Dockerfile's `publish-mlx` target.

The payload's `libmlx.so` reports `0.32.2-37-gce916db`, and its grammar library has all ten
symbols the loader looks up. Its llama-server binaries are 0.33.3's, built from the same
llama.cpp sources, so the preflight waits for the full image.

**Structured output works again, and ADR 0009 holds.** The same four requests as main's step 0,
on gemma4:12b-nvfp4:

| format | main | fold, rebuilt payload |
|---|---|---|
| `"json"` | 200, a JSON object | 200, the same object |
| a JSON Schema | 200, matching | 200, matching |
| `"yaml"` | 400, "invalid format: expected "json" or a valid JSON Schema object" | 400, "invalid structured output grammar: compile grammar: Check failed: … Invalid structural tag error: JSON schema format must have a json_schema field with a object" |
| `""` | 200, unconstrained | 200, unconstrained |

The `"yaml"` answer is still an error before any output. But the message is now xgrammar's
internal check failure instead of main's API-level explanation, because the check moved from
`parseGrammar` into the native compile. That is worse for a caller, and a friendlier message in
`requestGrammar` or `parseGrammar` would restore it.

The runner also logged one xgrammar warning at the end of the schema request, whose output was
correct: `grammar_matcher.cc:612: Warning: The matcher has terminated after accepting the stop
token, but is tr…`. The gate script truncates log lines at 200 characters and the container is
gone, so the rest of the line is lost. It suggests one token was offered to a matcher that had
already terminated, near the new rollback path for speculative decoding under a grammar. The
campaign's runner log is captured in full (`preflight-runs/sync034_thinkfalse-runner.log`) to see
whether it recurs.

**The idle core is fixed.** `claude-scratch/probe-idle-cpu.sh` sends one 8-token request, then no
traffic, and reads runner CPU from `/proc` deltas:

| t after the request | main, 2026-09-10 | fold, 2026-09-12 |
|---|---|---|
| +10 s | 100.2 % | 0.1 % |
| +20 s | 100.0 % | 0.0 % |
| +30 s | not sampled | 0.1 % |
| +40 s | 100.1 % | 0.1 % |
| +50 s | not sampled | 0.1 % |
| +60 s | 100.1 % | 1.4 % |

The GPU column in the fold's log shows the teacher leg on the same card, not the idle runner.

## Gate 3: why the Go-only swap covers only the GGUF half

The MLX, MLX-C and llama.cpp pins are byte-identical to v0.33.3. But upstream changed the xgrammar
shim that this repo builds itself: `x/mlxrunner/xgrammar/native`, which ships in the MLX payload
as `libollama_xgrammar.so`. The 0.34.0 loader (`x/mlxrunner/xgrammar/dynamic.c`) looks up ten
symbols in that library. Two are missing from the 0.33.3 build, both for speculative decoding
under a grammar: `ollama_xgrammar_matcher_rollback` and `ollama_xgrammar_matcher_is_terminated`.
The rebuilt payload has all ten. An earlier version of this section said five. The other three
names it listed, `ollama_xgrammar_dynamic_matcher_*`, are the loader's own wrapper functions,
defined in `dynamic.c`, not symbols it needs from the library.

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
