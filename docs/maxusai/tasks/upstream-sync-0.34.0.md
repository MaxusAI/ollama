# TASK: fold upstream v0.34.0 into main

Branch `task/upstream-sync-0.34.0` (worktree `claude-scratch/wt-sync034`), cut from `main` at
`7d230fdc` (#293). Upstream v0.34.0 (`d8ab4b4f`, released 2026-09-05) adds 21 commits across
88 files. They are mostly `app/`, `cmd/`, `server/` and `x/mlxrunner`, where the changes are
structured output compiled as xgrammar structural tags and speculative decoding under a grammar.
First look: [upstream-sync-2026-09-04.md](upstream-sync-2026-09-04.md), section "Next fold".
**PR #297**, opened 2026-09-13 against `main`. Not merged; the deploy stays held.

**The MLX bump is in.** `fbedf506` pins MLX to `ce916dbb`, which is ml-explore/mlx#4452, the fix
for the idle runner that pins a CPU core. The decision is below (2026-09-11 23:30).

## Status (2026-09-12, 06:40)

| gate | state |
|---|---|
| 1, merge | done, `ca3ff1db`, three conflicts resolved |
| 2, no-GPU tests | green after two fixes in `ba2eb4f1`; one upstream flake, two findings already on main |
| 3, image | **built**: `maxusai/ollama:sync-0.34.0` (`0.33.3-dynres-28-gfbedf50`, 05:05). Its MLX payload is byte-identical to the payload-swap image's (2,648 files), and its Go code matches, so the MLX gates measured there cover it |
| 4, preflight `cuda-dynres-903` | PASS 20, SKIP 8 on the swap image, the same as main; on the real image **PASS**, and 21/7 once the MLX pin was added and made assertable (below) |
| 5, GGUF think-off against `ggmlmain_1_` | **green**: 8 suites, no OOM, no error; every quality row identical to main |
| MLX format check and idle-CPU probe | **passed** on the rebuilt MLX payload (below) |
| 5, MLX think-off against `main276_` | done on the payload-swap image, 01:33 to 02:58: 5 suites, no OOM, no error. The few moved cells are knife-edge flips that drafting under a grammar adds (attribution below). **Repeated on the real image** 23:28 to 00:12 (`cand034_`): contract matrix identical to main, 2 of 70 quality cells differ |
| MLX attribution, gates 034e to 034i | **done**. Without drafting under a grammar the fold matches main on outputs and memory. With it, knife-edge cells flip between runs and qwen3.5-family memory grows untracked across requests — which main's own drafting does too. **Decided** (below): keep upstream's default, `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0` restores ours |

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

### Attribution: gates 034e to 034i

The render is `preflight-runs/specab-render.md`, from `claude-scratch/render-specab.sh`. Every table in it is
generator output. The images:

- **fold:** `sync-0.34.0-mlxswap`.
- **nodraft:** the fold with main's one-line gate restored, `sync-0.34.0-nodraft`, built from a detached worktree and
  never committed.
- **main:** `main-a523d60b`.
- **mainmlx:** main with only the fold's `libmlx.so` and `libmlxc.so`, `main-a523d60b-mlxce916`. It separates the MLX
  bump from every Go change.

The fold, nodraft and mainmlx images were removed from Docker at 06:26 to spare the full root disk. Each can be
rebuilt in about a minute, with no CUDA compile, from `main-a523d60b` and the exports on the 8TB array:
`mlxpayload-034`, `gobin-034b` and `gobin-034-nodraft`.

**Cold repeats (034e, 034f).** Each image ran the arms that moved five times, with a cold restart before every run.

- **gemma4:12b `bbox_contract_adv_norm1`:** the fold, nodraft and main each give the same answer on all five runs,
  contract ❌. mainmlx ran only qwen3.6 cold.
- **qwen3.6:** the same knife-edge cells flip on every image except mainmlx, which gave one answer all five times.
  Main's document name_bbox lands on 0.504 or 0.613, and the 7 px tier on the fold and on nodraft lands on 1 or 2.

So on cold loads neither drafting nor the MLX bump moves these arms beyond the flips each image shows on its own.

One caveat bounds that result: **cold, the fold barely drafts.** Its depth controller starts at depth 0 after every
load. It drafts deep only once its cost model holds a clean sample at two depths, and here that took two requests. A
cold single-request repeat drafted about 0.02 tokens per round. A full suite drafts from its third request on, about
3 per round on qwen3.6.

**Full-suite repeats (034g).** Main and the fold each ran the full suite twice more.

- **Main** gives contract ✅ on bcadvnorm1 in all three of its full-suite runs, and ❌ in all five cold ones. The
  answer depends on context but not on the run. In a full suite the test restores a cached prefix from the test before
  it, which sends the same three images.
- **The fold** drafts warm in a full suite, and gives ❌, ❌, then ✅.
- **qwen3.6's name_bbox** varies on both images.

Three runs a side is suggestive, not conclusive. Answer length, measured by the inspection script
`claude-scratch/evaldiff.py`, shows how far apart two runs of the same image already are:

- **gemma4:12b:** main against main differs on 2 of 27 tests, the fold against itself on 3, and main against the fold
  on 2 to 3.
- **qwen3.6:** main against main differs on 7 tests, the fold against itself on 7, and main against the fold on 7 to 8.

The gate's single-run comparison sits inside each image's own variation.

**Drafting or the MLX bump (034i).** nodraft is the fold with drafting under a grammar switched off. It keeps the
same MLX payload and every other v0.34.0 Go change.

- **bcadvnorm1:** in a full suite nodraft gives main's answer, contract ✅.
- **Length:** evaldiff shows nodraft's answer lengths matching main's first full-suite repeat on all 27 tests. Where it
  differs from main's second repeat, it's the same two tests on which main's two repeats differ from each other.

So without drafting, the fold lands inside main's own run-to-run variation, and the MLX bump and the rest of the fold
leave these outputs alone. mainmlx confirms it from the other side. With main's Go and only the fold's MLX
libraries, its full suite gives main's answer, contract ✅, and its answer lengths match main's first full-suite
repeat, and nodraft's, on all 27 tests.

What remains is drafting under a grammar. The runs without it give ✅ on bcadvnorm1 in 4 of 4 full suites; the runs
with it give ✅ in 1 of 3. That is weak evidence on its own (Fisher's exact test, p ≈ 0.14), but the mechanism fits:

- Verification runs the target over the drafts in one batch, so its logits can pick a different argmax than one-token
  decode at a near-tie.
- The depth controller is timing-driven, so where drafting happens changes from run to run.
- Against nodraft, each fold run differs in answer length on one or two extra gemma4 tests.

**Memory: drafting under a grammar leaves untracked MLX memory behind on qwen3.8.** Gate 034i ran the qwen3.8 full
suite at trace level on nodraft and on the fold. After each request's teardown the runner lists every live array it
tracks, next to MLX's own active-memory figure. The numbers come from `summarize_retained_memory.py` and
`summarize_peak_memory.py`, promoted into the suite in #295.

- **The prefix-cache trie holds the same content on both, request for request.** The recurrent snapshots grow by 144
  arrays (48 layers × 3 snapshots) per request up to the 8 GiB cap, and the paged-out bytes match.
- **On nodraft, MLX's active memory matches the tracked arrays** within 0.14 GiB all suite long.
- **On the fold, active memory runs ahead of the tracked arrays, and the gap grows:** +0.7 GiB after request 4, +3.7
  after 13, +5.8 after 22, +6.8 after 28, with no plateau. The memory survives the teardown's sweep and cache clear
  but belongs to no array the runner tracks, so neither the admission headroom nor the trie's cap sees it.
- **Per-request peaks:** nodraft matches main (median +0.00 GiB), so the MLX bump and the other v0.34.0 changes cost
  nothing. The fold runs a median 2.7 GiB above nodraft, and up to 7.0 GiB, at a maximum draft depth of 6. The 01:33
  run drafted up to 16 deep and ran up to 14 GiB above main.
- **gemma4's peaks match main's** at every draft depth.

**Main leaks too, so this is not new in v0.34.0.** `claude-scratch/probe-mainleak.sh` sent the same 18 text
requests to `main-a523d60b` twice on qwen3.8: once without a format, where main drafts, and once with one, where it
does not. Drafting followed the format exactly — 18 of 18 completions drafted without it, 0 of 18 with it — and only
the drafting arm grows:

| request | main drafting | main not drafting |
|---|---|---|
| 1 | −0.14 GiB | −0.14 GiB |
| 10 | +0.02 | −0.13 |
| 13 | +0.44 | −0.13 |
| 17 | +0.72 | −0.11 |

Peaks: median +0.58 GiB, largest +1.26, at shallow depth on short answers. Smaller than the fold's figures because
this workload drafts shallow and the prompts are short, not because the path differs.

So the leak lives in the speculation path main already ships, and production meets it today on think-on and
format-less requests. What v0.34.0 changes is reach: drafting under a grammar extends it to structured output, which
is nearly all of this suite's traffic. **Restoring main's gate limits the leak; it does not remove it.**

The mechanism is not pinned down, so the fix is not ours to propose yet. `KVCache`'s lazy snapshots hold no handle on
the live buffer by design, so they are not it. The likely candidates are lazily built arrays that the speculation
path keeps across rounds, for example the draft side holding slices of the verification forward's hidden states.

**Rendered evidence.** These are generator tables from `preflight-runs/specab-render.md`, pasted verbatim. The
render also holds every cold run's contract matrix and the T2 pivots.

qwen3.6 cold repeats, from `summarize_reps.py`:

| metric | main 09-06 (n=1) | fold 09-12 (n=1) | fold (n=5) | nodraft (n=5) | main (n=5) | mainmlx (n=5) |
|---|---|---|---|---|---|---|
| **num_ctx rung** | 8192 | 8192 | 8192 | 8192 | 8192 | 8192 |
| **num_predict** | 2200 | 2200 | 2200 | 2200 | 2200 | 2200 |
| scene bbox IoU | 0.964 | 0.965 | — | — | — | — |
| scene labels | 6 | 6 | — | — | — | — |
| scene colors | 6 | 6 | — | — | — | — |
| scene serial | 1/1 ✅ | 1/1 ✅ | — | — | — | — |
| doc items | 5 | 5 | 5 | 5 | 5 | 5 |
| doc qty+price | 5 | 5 | 5 | 5 | 5 | 5 |
| doc total | 1/1 ✅ | 1/1 ✅ | 5/5 ✅ | 5/5 ✅ | 5/5 ✅ | 5/5 ✅ |
| doc name_bbox IoU | 0.506 | 0.613 | 0.504 | 0.504 | 0.548 [0.504–0.613] | 0.504 |
| multi q1 | 1/1 ✅ | 1/1 ✅ | — | — | — | — |
| multi q2 | 1/1 ✅ | 1/1 ✅ | — | — | — | — |
| multi q4-bbox | 1/1 ✅ | 1/1 ✅ | — | — | — | — |
| multi chart | 5 | 5 | — | — | — | — |
| finetext 22px | 4 | 4 | 4 | 4 | 4 | 4 |
| finetext 16px | 4 | 4 | 4 | 4 | 4 | 4 |
| finetext 12px | 4 | 4 | 4 | 4 | 4 | 4 |
| finetext 9px | 2 | 2 | 2 | 2 | 2 | 2 |
| finetext 7px | 1 | 2 | 1.6 [1–2] | 1.2 [1–2] | 1 | 1 |
| finetext correct /20 | 15 | 16 | 15.6 [15–16] | 15.2 [15–16] | 15 | 15 |
| finetext unmatched | 5 | 4 | 4.4 [4–5] | 4.8 [4–5] | 5 | 5 |

```text
Within-arm spread (max-min), the bar any cross-arm claim must clear:
  main 09-06: n=1, no spread measurable
  fold 09-12: n=1, no spread measurable
  fold: counts — finetext unmatched 1, finetext correct /20 1, finetext 7px 1
  nodraft: counts — finetext unmatched 1, finetext correct /20 1, finetext 7px 1
  main: ratios — doc name_bbox IoU 0.109
  mainmlx: identical across all 5 runs
```

qwen3.6 full suites, from `summarize_reps.py`:

| metric | main 09-06 (n=1) | fold 09-12 (n=1) | main full (n=2) | fold full (n=2) |
|---|---|---|---|---|
| **num_ctx rung** | 8192 | 8192 | 8192 | 8192 |
| **num_predict** | 2200 | 2200 | 2200 | 2200 |
| scene bbox IoU | 0.964 | 0.965 | 0.964 | 0.964 |
| scene labels | 6 | 6 | 6 | 6 |
| scene colors | 6 | 6 | 6 | 6 |
| scene serial | 1/1 ✅ | 1/1 ✅ | 2/2 ✅ | 2/2 ✅ |
| doc items | 5 | 5 | 5 | 5 |
| doc qty+price | 5 | 5 | 5 | 5 |
| doc total | 1/1 ✅ | 1/1 ✅ | 2/2 ✅ | 2/2 ✅ |
| doc name_bbox IoU | 0.506 | 0.613 | 0.558 [0.504–0.613] | 0.558 [0.504–0.613] |
| multi q1 | 1/1 ✅ | 1/1 ✅ | 2/2 ✅ | 2/2 ✅ |
| multi q2 | 1/1 ✅ | 1/1 ✅ | 2/2 ✅ | 2/2 ✅ |
| multi q4-bbox | 1/1 ✅ | 1/1 ✅ | 2/2 ✅ | 2/2 ✅ |
| multi chart | 5 | 5 | 5 | 5 |
| finetext 22px | 4 | 4 | 4 | 4 |
| finetext 16px | 4 | 4 | 4 | 4 |
| finetext 12px | 4 | 4 | 4 | 4 |
| finetext 9px | 2 | 2 | 2 | 2 |
| finetext 7px | 1 | 2 | 1 | 1.5 [1–2] |
| finetext correct /20 | 15 | 16 | 15 | 15.5 [15–16] |
| finetext unmatched | 5 | 4 | 5 | 4.5 [4–5] |

```text
Within-arm spread (max-min), the bar any cross-arm claim must clear:
  main 09-06: n=1, no spread measurable
  fold 09-12: n=1, no spread measurable
  main full: ratios — doc name_bbox IoU 0.109
  fold full: ratios — doc name_bbox IoU 0.109; counts — finetext unmatched 1, finetext correct /20 1, finetext 7px 1
```

gemma4:12b full suites, from `summarize_contract_matrix.py`, one table per run; bcadvnorm1 is the `bcadvnorm1` column:

`main276_1_`, main, the gate's run (2026-09-06):

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |

`sync034_1_`, fold, the gate's run (2026-09-12 01:33):

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | 8192 |

`specfull_main_1_`, main, full-suite repeat 1:

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |

`specfull_main_2_`, main, full-suite repeat 2:

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |

`specfull_fold_1_`, fold, full-suite repeat 1:

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | 8192 |

`specfull_fold_2_`, fold, full-suite repeat 2:

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |

`specmem_nodraft_1_`, nodraft, full suite:

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |

`specmem_mainmlx_1_`, mainmlx, full suite:

| Model | Engine | bc | bcmulti | bcreasoning | bcpinned | bcperobject | bcanchored | bcadvreal | bcadvnorm1 | num_ctx |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ | 8192 |

**What this means for the gates.** ADR 0012 §4 says think-off quality cells are bit-reproducible per (payload,
backend, budget, image). On MLX they are not:

- they move with the prefix-cache state a test inherits;
- knife-edge cells flip across cold loads;
- warm, timing-driven drafting adds more flips.

`vision-lowtemp-thinkon-negative-result.md` already calls MLX temperature 0 non-reproducible across loads, but §4 was
never amended. So a single-run MLX gate cannot tell a near-tie flip from a real change, and a cross-build difference
needs repeats on both builds, three or more as here. Amending §4 is Glenn's call.

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

## The deploy candidate: `maxusai/ollama:sync-0.34.0` (gate-sync034h)

Built from `fbedf506` at 05:05 (`0.33.3-dynres-28-gfbedf50`, 5.46 GB). Two proofs tie it to the payload-swap image
the MLX gates ran on:

- **P1:** its MLX payload is byte-identical, file by file (2,648 files, sha256).
- **P2:** `fbedf506` and `5404bec2`, the swap image's binary, differ only under `docs/`.

So everything measured on the swap image holds for it: structured output, the idle core, the think-off campaign and
the attribution above. The live checks repeat on the real image:

- **Formats:** `"json"` gives 200 with the same object; a JSON Schema gives 200, matching; `"yaml"` gives 400 with
  xgrammar's structural-tag error; `""` gives 200.
- **Idle core:** the idle runner holds 0.0 to 0.2 % CPU.
- **MLX pin:** the profile now records `mlx_build = ce916dbb`, this fold's `MLX_VERSION`. It recorded none until
  2026-09-12, so `mlx_payload_pin` skipped on every CUDA run and this bump went unasserted; PR #294 adds the same key
  on main with `37c26e57`.

  Asserting it needed a second source. The check reads the MLX build from the runner's engine-init line, and a CUDA
  preflight loads llama.cpp models, so no MLX runner ever starts: pinning alone turned the skip into a failed run,
  `no MLX engine-init line in the log window`, on a payload that was exactly the pinned one. `probes.mlx_build_payload`
  now reads the version string out of the shipped `libmlx.so`, the way `llama_cpp_build` reads `llama-server`. The
  live line stays preferred, because only a load catches binary/payload skew, and the result says which source
  answered.
- **Preflight `cuda-dynres-903`:** PASS. The 06:40 run was 20 pass, 8 skip, with the MLX pin among the skips
  (`vision-suite/preflight/runs/full-sync034-real.json`). Re-run at 21:36 with the pin in place: **21 pass, 7 skip**,
  `mlx_payload_pin: MLX build ce916db matches the measured payload (0.32.2-37-gce916db), from the shipped libmlx.so`
  (`preflight-runs/full-sync034-pinned.json`).

**Its own think-off campaign** ran 2026-09-12 23:28 to 00:12 (`claude-scratch/gate-sync034j.sh`, tag `cand034_`,
rendered to `preflight-runs/cand034-render.md`). Until then the real image was covered by P1 and P2 rather than by a
campaign of its own. Five models, no OOM, no error, nothing left unconverged:

- **Contract matrix: identical to main, 50 of 50 cells**, `bbox_contract_adv_norm1` included — the knife-edge cell
  that came back ❌ on the swap image's run and ✅ here, which is what a knife-edge does.
- **T1 quality: 2 of 70 cells differ from either baseline.** Against the swap image: gemma4:26b's scene IoU 0.972 to
  0.969, and qwen3.6's 7 px tier 2 to 1. Against main: that same IoU, and qwen3.6's scene IoU 0.964 to 0.965.
- **Answer lengths** differ from the swap image on 2 to 8 of 27 tests per model, and from main on 1 to 9 — inside
  the spread two runs of one image already show (`summarize_output_lengths.py`).
- **It drafts as upstream intends:** every one of 28 completions per model, 3.2 to 4.7 tokens per target forward on
  the four larger models. gemma4:12b drafted almost nothing this time, 0.003 per round against 0.443 on the earlier
  campaign, which is the timing-driven controller rather than a build difference.

So the artifact that would ship behaves like the image the fold was measured on, and like main.

The deploy stays held for Glenn. It also carries the decision below on drafting under a grammar.

## Decision: drafting under a grammar stays on, ours is a knob (Glenn, 2026-09-12)

Upstream `4986e923` lets a structured-output request draft; main's `speculation.open` refused to. Everything else in
the fold matches main on these gates — nodraft, the fold with main's one line restored, matches main on outputs and
on memory — so the choice came down to that one line.

**Decided: align with upstream.** The fold keeps upstream's behaviour as the default, and
`OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0` restores main's gate for an operator who wants it. `speculate.go` reads the knob
once at startup through `draftingEnabled`; an unrecognised value keeps the default and warns, so a typo cannot
quietly halve structured-output throughput. Nothing goes to `ollama/ollama`.

What that buys and costs, measured:

- **For:** 1.5 to 2.6 times the generation rate on structured output for the four larger models, in the full-suite
  context. Both runs sat beside production and the MLX pins differed, so read the ratio as context, not a number.
- **Against, outputs:** knife-edge answers flip between runs. They sit inside the spread MLX already has — two runs
  of main differ in answer length on 2 of 27 gemma4 tests and 7 of 27 qwen3.6 tests — and no quality row moved.
- **Against, memory:** drafting leaves MLX memory that no tracked array accounts for, growing across requests
  (+6.8 GiB by request 28 on qwen3.8), which the admission headroom cannot see. **Not new in v0.34.0:** main's own
  drafting does the same on think-on and format-less requests; drafting under a grammar widens the reach to
  structured output.

**What to watch after a deploy.** A long-running MLX server on the qwen3.5 family under sustained structured-output
load: its memory climbs past what admission priced, so a co-resident load can be admitted into memory the runner
will need. The knob is the mitigation until the leak is fixed upstream. ADR 0033 records the divergence and ADR 0034
records that the headroom cannot price it.

**The knob is verified on a real image** (`claude-scratch/probe-knob.sh`, 22:43). `maxusai/ollama:sync-0.34.0-knob`
is the deploy candidate with this branch's Go binary layered on — `0.33.3-dynres-41-gfbe18d0`, MLX payload
byte-identical to the candidate's, binary sha checked against the build. A parked session logs no draft-stats line,
so counting those lines per request type is the signal:

| `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR` | 4 requests carrying a format | 2 requests without one |
|---|---|---|
| unset, upstream's default | 4 drafted | 2 drafted |
| `0` | **0 drafted** | 2 drafted |
| `maybe`, unrecognised | 4 drafted, and the runner warns | not run |

So it gates structured output only, leaves ordinary requests drafting, and an unrecognised value keeps upstream's
default while saying so. Every request answered 200.

**The clean candidate does not carry it.** `maxusai/ollama:sync-0.34.0` was built before the knob, so it has
upstream's default and no way to turn it off; `-knob` is a binary swap for testing and for an operator who needs the
switch now. The knob belongs in the next full build.

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
