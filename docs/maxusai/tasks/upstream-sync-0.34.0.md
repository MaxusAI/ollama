# TASK: fold upstream v0.34.0 into main

Branch `task/upstream-sync-0.34.0` (worktree `claude-scratch/wt-sync034`), cut from `main` at
`7d230fdc` (#293). Upstream v0.34.0 (`d8ab4b4f`, released 2026-09-05) adds 21 commits across
88 files. They are mostly `app/`, `cmd/`, `server/` and `x/mlxrunner`, where the changes are
structured output compiled as xgrammar structural tags and speculative decoding under a grammar.
First look: [upstream-sync-2026-09-04.md](upstream-sync-2026-09-04.md), section "Next fold".

**The MLX bump is in.** `fbedf506` pins MLX to `ce916dbb`, which is ml-explore/mlx#4452, the fix
for the idle runner that pins a CPU core. The decision is below (2026-09-11 23:30).

## Status (2026-09-12, 03:25)

| gate | state |
|---|---|
| 1, merge | done, `ca3ff1db`, three conflicts resolved |
| 2, no-GPU tests | green after two fixes in `ba2eb4f1`; one upstream flake, two findings already on main |
| 3, image | Go-only swap `maxusai/ollama:sync-0.34.0-swap` (`0.33.3-dynres-26-gba2eb4f`), **valid for the GGUF half only**; payload swap `sync-0.34.0-mlxswap` for the MLX half; the full image `sync-0.34.0` is still building |
| 4, preflight `cuda-dynres-903` | PASS 20, SKIP 8 on the swap image, the same as main; repeated on the rebuilt image |
| 5, GGUF think-off against `ggmlmain_1_` | **green**: 8 suites, no OOM, no error; every quality row identical to main |
| MLX format check and idle-CPU probe | **passed** on the rebuilt MLX payload (below) |
| 5, MLX think-off against `main276_` | done on the payload-swap image, 01:33 to 02:58: 5 suites, no OOM, no error. Identical to main except a few cells, and the fold drafts under a grammar where main never did (below) |
| MLX attribution A/B, `gate-sync034e` | running since 03:13: does drafting under a grammar explain those cells? |

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
whether it recurs. It does, and it is benign. See the next section.

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

## MLX think-off against `main276_`: a few cells moved, and the fold now drafts

The five-model campaign (tag `sync034_1_`, 01:33 to 02:58) finished with no OOM, no error and no
cell left capped. Main's generators rendered it in `preflight-runs/sync034-render.md`
(`claude-scratch/render-sync034.sh`).

`claude-scratch/diff-sync034.sh` diffs the fold's tables against main's mechanically; its output
is `preflight-runs/sync034-diff.txt`. Every cell of T1, T2 and the contract matrix is identical
except these:

- **gemma4:12b-nvfp4, bcadvnorm1.** The contract went from ✅ to ❌.
- **qwen3.6:35b-a3b-nvfp4.**
  - document name_bbox IoU: 0.506 to 0.613;
  - fine text: 4/4/4/2/1 to 4/4/4/2/2;
  - T1 Answer tok (the length of the scene request): 537 to 549;
  - scene bbox IoU: 0.964 to 0.965.
- **gemma4:31b-nvfp4, document name_bbox IoU.** 0.751 to 0.750.

gemma4:26b and qwen3.8:27b match main in every cell.

**name_bbox is a knife-edge arm.** #287's control is acceptance criterion 2 in
`docs/maxusai/tasks/mlx-prefill-dequant-gemm.md` on `feat/mlx-prefill-dequant-gemm`, rendered with
`summarize_reps.py`. It found each numeric path bit-reproducible on this arm, and different paths
far apart: qwen3.6 scored 0.504 in three runs with that PR's flag off and 0.613 in three with it
on. The fold's 0.613 is one of the answers another path already produces, so on its own it says
nothing about quality.

**Every suite request carries a grammar, and only the fold drafts under one.** The suite's client
sends `format: "json"` by default, so every request is structured output. Main's
`speculation.open` refused to draft under a grammar (`enabled := request.Grammar == nil && …`).
Upstream `4986e923` ("mlxrunner: enable speculative decoding under structured output", Jesse Gross,
2026-08-28) dropped that condition and masks each draft position during verification instead, to
recover what the commit calls "roughly half the speculative throughput on a dense 27B MTP model".

The drafts come from each model's own head. gemma4 uses its bundled assistant model, and the qwen3.5
family (qwen3.6:35b-a3b included) uses its MTP head. The runner logs show the change:

- main's think-off log (`main276dq_1_`) has no speculative-stats line in 112 completions;
- the fold's log has one for every one of its 139.

An inspection script over the runner logs, `claude-scratch/draftstats.py`, puts numbers on it. It
is not a vision-suite generator, so these are inspection-grade:

- the four larger models commit 3.2 to 5.4 tokens per target forward: qwen3.8:27b 5.4,
  gemma4:26b 4.9, gemma4:31b 4.2, qwen3.6:35b-a3b 3.2;
- gemma4:12b commits 1.36, because its assistant is rarely worth running (0.44 drafted tokens per
  round).

At temperature 0 the verification is greedy, so drafting can change a token only where the
batched verification forward and the one-token forward pick a different argmax: a near-tie.

Scored cells understate how often the text changes. A second inspection script,
`claude-scratch/evaldiff.py`, compares each test's `eval_count` and `answer_chars` between the two
campaigns. Between main and the fold, the lengths differ in 2 to 7 of 27 tests on every model:

- qwen3.8:27b 2;
- gemma4:31b 3;
- gemma4:12b 4;
- gemma4:26b 6;
- qwen3.6:35b-a3b 7.

Equal lengths do not prove equal text, so these counts are a floor. Nor do they say whether
drafting or the MLX bump did it. The A/B below attributes the three arms whose scores moved, not
the rest.

**Throughput is why upstream made the change, and T1 points the same way here.** In
`preflight-runs/sync034-render.md`, T1's Gen tok/s rises from main to the fold on the four models
that draft well:

- gemma4:26b 32 to 57;
- gemma4:31b 23 to 34;
- qwen3.8:27b 16 to 42;
- qwen3.6:35b-a3b 19 to 40.

It falls on gemma4:12b, from 31 to 27. Both runs shared GPU0 with production, and the MLX pins
differ, so these figures are context, not a measurement.

**The xgrammar warning recurs and is benign.** 84 of the fold's 139 completions log "The matcher
has terminated after accepting the stop token, but is trying to accept new token". `grammarEngine.fill`
builds the per-position masks by accepting each draft into the matcher and rolling the walk back
afterwards. When a draft chain holds the stop token followed by more drafts, the terminated matcher
refuses the next one and xgrammar logs the warning. `accept` cuts the run at an accepted stop
token, so no position past it is ever committed.

**Attribution: `claude-scratch/gate-sync034e.sh`, running since 03:13.** It runs three images over
the three arms that moved, 5 repeats each:

- fold: `sync-0.34.0-mlxswap`;
- nodraft: `sync-0.34.0-nodraft`, the fold with main's gate restored. That is one line in
  `speculate.go`, built from a detached worktree at `2b5478ab` and never committed;
- main: `main-a523d60b`.

The arms are gemma4:12b-nvfp4 on `bbox_contract_adv_norm1`, and qwen3.6:35b-a3b-nvfp4 on
`document_single` and `finetext`. nodraft against the fold isolates drafting. nodraft against main
isolates the MLX bump and the rest of the fold. `claude-scratch/render-specab.sh` renders the result
once the gate is done.

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
