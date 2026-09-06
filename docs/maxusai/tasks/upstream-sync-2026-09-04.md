# TASK: fold upstream v0.33.3 into main — plan

**Opened:** 2026-09-04. **Status:** DECIDED 2026-09-04 (Glenn), housekeeping in
progress, no merge attempted on `main`. **D1 = keep `image_max_tokens`**: the
per-request image-token budget survives, so this fold keeps the fork's gemma4
MLX vision (option A below); adopting upstream's tower with the budget seam
re-grafted (option B) is a separate measured spike + ADR. **D2 = ADR 0010's
arithmetic on upstream's plumbing, aligned with upstream**: take
`IncludeIntermediateMetrics` / `firstPassMetrics` / `PromptEvalCachedCount` as
upstream shapes them, and keep the fork's cache-inclusive prompt-count
derivation only where the two disagree (vision requests). Everything else
below is mechanical and verified.

**Merged 2026-09-04** on `task/upstream-sync-0.33.3` (Gates 1 and 2 green,
criteria 3 and 4 below). One D2 divergence shipped and is worth naming here:
upstream's new restart test asserts *its* duration split — the continuation's
prefill reclassified as generation work — and this fold keeps ADR 0004's
summing instead, because adopting upstream's would move every recorded
`think + format` cell's tok/s. Counts, cached counts and eval counts fold
identically to upstream's; only the duration split differs. Assessment by two max-effort reviews (PR history
#212–#263; upstream delta) with every load-bearing claim re-verified against
the tree, the real llama.cpp checkouts at b10630/b10760, and the deployed
server.

## Scope

14 upstream commits, `v0.33.2..v0.33.3` (released 2026-09-02); 156 files,
+12,844 / −6,387. Our sync point is `v0.33.2-dynres` (`eaaf9518`, #232);
`main` is `v0.33.2-dynres-32-gb54d4d0d`. A dry `git merge-tree --write-tree
origin/main v0.33.3` produces **28 conflicts**, five of them add/add.

**Both payload pins move — and MLX-C too:**

| pin | now | v0.33.3 |
|---|---|---|
| `LLAMA_CPP_VERSION` | `b10630` (`d222767c7`) | **`b10760`** (`0f3a71be1`) |
| `MLX_VERSION` | `c793734eb715…` | **`37c26e5755da…`** |
| `MLX_C_VERSION` | `fba4470b8907…` | **`c74db5307cc8…`** (regenerated bindings; `mlx/compat/` carry patch deleted) |

Toolchain does not move: Go 1.26.0, CUDA 12.8 / 13.0, ROCm 7.2.1, CMake
3.31.2, Ninja 1.12.1. Only `go.mod` change is an indirect `go-localereader`
bump.

| commit | what | our exposure |
|---|---|---|
| `b79067b0` gemma4: image and audio input support | upstream's own gemma4 MLX vision + audio | **collides add/add with our ADR 0021 vision** → D1 |
| `855f4bf9` Report cached prompt tokens | `prompt_eval_cached_count` (additive); `IncludeIntermediateMetrics` in the two-pass flow | `routes.go` / `pipeline.go` / `client.go` / `llama_server.go` conflicts → D2 |
| `f348c7e3` Honor model generation defaults | GGUF `general.sampling.*` applied live at load | **verified inert on our fleet** (below) |
| `ba064c36`, `c36adebc` mlxrunner: check every mlx-c call / single error buffer | `mlxCall` and the `__thread` error buffer deleted | our `ClaimOSThread` (ADR 0017) and `memory.go` sit on `mlxCall` |
| `e5e43771` ci: MLX unit tests for PR runs | `mlxtest.Setup` → `mlxtest.Run` | 18 call sites in 10 fork files, incl. the vision goldens |
| `3ffc9a68` shared audio decoding | new `x/mlxrunner/model/audio/` | clean add |
| `5ec58043`, `b1d1ccc9` llama.cpp b10729, b10760 | vendor bump; upstream re-cut its `001` hooks patch | compat band verified (below) |
| `3ba380d0` MLX, MLX-C bump; `ef117cfc` dedup deps; `205a0426` go-license step | build/packaging | `cmake/mlx` conflict (one line) |
| `882387a5`, `e37a00a8` | test fix, typos | none |

## What we gain

Upstream-maintained gemma4 MLX vision **and audio** (if D1 adopts it), the
cached-prompt-token metric, MLX-C error checking on every call (the class of
masked error #212's `guardClose` had to work around), llama.cpp b10760 (130
commits), the MLX/MLX-C bump, and CI that runs MLX unit tests on PRs.

## Verified facts

- **Compat band at b10760 — clean.** On a real `b10760` checkout, upstream's
  re-cut `001-llama-cpp-hooks.patch` (from v0.33.3) followed by our
  `002/004/005/801/903` all `git apply` in sequence. Our *copy* of `001` does
  not apply at b10760 — that is expected; the fold takes upstream's file.
  `tools/mtmd/clip.cpp` moved by one unrelated hunk (qwen3-tts optional
  tensors); `mtmd-image.cpp`, `clip-model.h`, `mtmd.cpp` are byte-identical.
- **903 stays.** ggml-org/llama.cpp#27044 is OPEN (2026-09-02). The only
  `mmq.cu` change (`ggml_cuda_should_use_mmq`, L317) routes MoE through MMQ on
  **pre-DP4A cards** — outside 903's hunk, no effect on Blackwell, but it
  widens 903's blast radius on older GPUs. Functional gate at the new pin
  before any pre-Ada deploy.
- **#215's gate stays valid.** `GGML_CUDA_CUBLAS_COMPUTE_TYPE` exists in
  b10760's `ggml-cuda.cu`; ollama/ollama#18070 is OPEN. `applyArchServerEnvs`
  lives in `startProcess()`, which upstream does not touch.
- **`prompt_eval_count` semantics unchanged.** MLX still reports
  `len(request.Tokens)`; the cached count is a *separate* field
  (`cachedPromptCount = len(session.inputs) − len(session.remaining)`).
  llama-server emits `cache_n` at both builds, so the new nil-guard never
  fires. Ladders are not expected to move on this account.
- **GGUF generation defaults are a no-op here.** `/api/show` on `:11497`:
  only `qwen3.8:27b-q4_K_M` carries `general.sampling.*` (temp/top_k/top_p),
  and its params blob already sets them at higher precedence. Re-run this
  check after any new `ollama pull`.
- **Four compile breaks hide behind clean auto-merges:** `x/mlxrunner/mlx/memory.go`
  (`mlxCall` ×3 — the VRAM/cache ceiling), `ClaimOSThread()` callers
  (`x/create/mlxthread.go:32`, `x/mlxrunner/server.go:43`, `mlxtest.go:47`),
  `mlxtest.Setup` (18 sites / 10 files, incl. `vision_golden_test.go`,
  `vision_e2e_test.go`, `constrain_bench_test.go`), and `isGemma4Renderer`
  (`server/images.go:487`, `server/model_list_cache.go:410`).
- **Preflight gate asymmetry.** `cuda-dynres-903`'s `version_pattern` already
  admits `0.33.3-dynres-*`, so on CUDA only `payload_pin` (`d222767c7`)
  stands between a b10760 payload and b10630 expectations. `mlx-metal-0-33-2`
  will refuse `0.33.3-maxusai-*` (exit 2) — correct; a new profile is
  required (ADR 0011 rule 5, #243 precedent).

## Conflicts: 28 files

All 28 resolved in the merge commit on this branch (`git merge --no-ff
v0.33.3`, second parent `b79067b0`). "Resolved as" records what was actually
done, not what was planned.

| file(s) | planned resolution | resolved as |
|---|---|---|
| `x/models/gemma4/{media,vision,media_test,vision_test}.go` (add/add), `gemma4.go`, `gemma4_test.go`, `gemma4_moe_test.go` | **D1**, whole-file per side; never hand-merged | **D1-A.** The four add/add files: ours whole-file. `gemma4.go`: ours, taking only upstream's package doc line — every other upstream hunk is vision/audio wiring (`parseMultimodalConfig`, `buildMasks` threaded through `DecoderLayer`/`Attention`, media-placeholder embed masking, `loadAudioWeights`) and does not compile against our tower. `gemma4_test.go` / `gemma4_moe_test.go`: theirs (pure `mlxtest` port, no vision content); `vision_test.go`'s nine `useMLXTestThread` sites ported to `mlxtest.Run`, since upstream deleted that helper along with `gemma4_moe_test.go`. Exclusions below. |
| `server/routes.go` (old 2747–2801), `x/mlxrunner/pipeline.go`, `x/mlxrunner/client.go`, `llm/llama_server.go` | **D2**; preserve `guardClose` call sites, the `stopper` block, `applyCompletionFormat`, `visionServerArgs`, `kvCacheFlagValues`, `ggmlCublasComputeTypeEnv`, `mlxRunnerEnvDefaults` | **D2.** Upstream's plumbing kept whole: `IncludeIntermediateMetrics` end to end, `includeIntermediateMetrics := req.Format != nil && currentFormat == nil`, the `firstPassMetrics` capture with its non-terminal blanking, `PromptEvalCachedCount` everywhere, and `pipeline.go`'s `cachedPromptCount = len(session.inputs) - len(session.remaining)` plus per-chunk metric enrichment. Upstream's `else if Applying && r.Done` fold is dropped — ADR 0004's `pass1` summing does the same job and both would count pass one twice. New `reportedPassMetrics()` prefers the runner's own pass-one report over ADR 0010's textual reconstruction at the transition site (that report is the cache-inclusive prefill, image tokens included); `transitionPassMetrics()`/`transitionPromptDelta` stay as the fallback, and the delta is still computed only from a reconstructed pass. #238's `deferring` gate kept on both sites. Every named symbol preserved; all `guardClose` call sites and the `stopper` block intact. |
| `x/mlxrunner/mlx/{mlx,stream}.go`, `mlx/thread_test.go` | take theirs; re-express ADR 0017's guarantee on `x/internal/mlxthread` (upstream's own answer); port `memory.go` off `mlxCall` onto `mlxError`; confirm `mlx_set_cache_limit` / `mlx_set_memory_limit` / `mlx_get_memory_limit` survive the MLX-C regeneration | Theirs. `ClaimOSThread` and `__thread _mlx_thread_owned` deleted with their four callers (`x/create/mlxthread.go`, `x/mlxrunner/server.go`, `vision_golden_test.go`, `vision_e2e_test.go`); ADR 0017 carries a status amendment and AGENTS.md / `docs/development.md` now point at `mlxthread` and `mlxtest.Run`. **All three MLX-C symbols confirmed present** with unchanged signatures (`x/mlxrunner/mlx/generated.h:5301-5316`, `include/mlx/c/memory.h:34-38`); `memory.go`'s three functions ported onto `mlxError`. `memory_test.go`'s fork-only `SetCacheLimit` test ported onto upstream's `withMLXThread(t, func(*mlxthreadtest.T))`. |
| `x/internal/mlxtest/mlxtest.go`, `x/mlxrunner/cache/*_test.go`, `model/embedding_test.go`, `sample/sample_test.go`, `x/models/{laguna,qwen3_5}/*_test.go`, `x/mlxrunner/client_test.go` (add/add) | take theirs; port the 18 `Setup` sites to `Run`/`RunSubtest`; concatenate the two `client_test.go` | Theirs; the two `client_test.go` concatenated (one package clause, union of imports, both bodies). 17 of the 18 `Setup` sites turned out to sit in upstream-owned files and upstream ported them itself. The 18th, `constrain_bench_test.go`, cannot be ported — it calls `mlxtest.Setup(b)` and the new API takes only `*testing.T` — so it and `constrain_test.go` (whose `skipIfNoMLX` came from an upstream file that no longer defines it) are **deleted**, bringing forward part of ADR 0033's follow-up. `constrain.go` and the four non-MLX `constrain_*_test.go` are left for that PR. |
| `server/images.go`, `server/images_test.go`, `server/model_list_cache.go` | take upstream's deletion of the gemma4 capability suppression; resolve `isGemma4Renderer` consistently with D1 (restore it under D1-A; drop the branch under D1-B) | Upstream's removal of the gemma4 **vision** suppression taken. `isGemma4Renderer` restored in `server/renderer_resolution.go` (D1-A). **Hidden break caught:** the auto-merge had silently taken upstream's deletion of the gemma4 branch in `suppressAudioCapability` too, which would have advertised an audio modality this fork cannot serve; restored. `model_list_cache.go`'s mirror kept; `images_test.go` keeps the fork's "keeps vision, suppresses audio" cases. |
| `cmake/mlx/CMakeLists.txt` | union: keep our `$ORIGIN` RPATH block and **`quadmath`**, take `cusolver cusparse nv[Jj]it[Ll]ink`, `OLLAMA_LIB_DIR`, license installs. Losing `quadmath` = a CUDA MLX payload that fails `CheckInit()` | Union exactly as planned; `quadmath` retained. |
| `.github/workflows/test.yaml` | union of path filters (as #232) + upstream's MLX unit-test job | Union: fork path filters + upstream's `go_mod_changed` filter, `go_license` job, `race` job and the MLX Darwin payload cache/prepare steps. |
| `docs/api.md` | take theirs, re-add our option docs | Theirs for `prompt_eval_cached_count` and the reworded `prompt_eval_duration`; the fork's `eval_count` note kept. Nothing to re-add — `image_min_tokens` / `image_max_tokens` and the kv-cache-type docs live under `docs/maxusai/`, never in `docs/api.md`. |

### Excluded under D1-A

Upstream's new gemma4 files that reference media.go symbols this fork does not
define (`multimodalConfig`, `MultimodalEmbedder`, `makeClippableLinear`,
`m.MM` / `m.Vision` / `m.Audio`, `visionSoftTokenBudget`) and therefore do not
compile against our package. The D1-B spike starts from this list:

- `x/models/gemma4/audio.go` — the conformer audio tower and its weight
  loading (`loadAudioWeights`, `encodeAudio`, `audioAttentionMask`).
- `x/models/gemma4/audio_test.go` — `prepareAudioMedia`, `parseAudioConfig`
  and the attention-mask tests for it.
- `x/models/gemma4/process_image.go` — upstream's image preprocessing:
  `visionTargetSize`, `preprocessImage`, `patchify`, `ImageGeometry`. This is
  the direct competitor to ADR 0008's `BudgetFillSize` ladder, and the file
  the B spike has to reconcile.

**Also excluded under D1-A (2026-09-04, after Gate 2):** `x/models/gemma4/process_audio.go`
and `process_audio_test.go`. They were first kept because they compile standalone
against the new `x/mlxrunner/model/audio` package, but golangci-lint's `unused`
flagged `processUnifiedAudio` at repo root — dead code that trips the lint gate is
worse than a file the D1-B spike can restore from the tag. The D1-B spike's
starting set is therefore all five upstream files: `audio.go`, `audio_test.go`,
`process_audio.go`, `process_audio_test.go`, `process_image.go`.

Also removed with `ClaimOSThread`: `TestMLXOperationsSurviveRescheduling`
(ADR 0017's conformance test) went with `mlx/thread_test.go`. It existed to
drive MLX from an *unpinned* goroutine, which upstream's shared-thread model
makes invalid; the contract is now pinned by `x/internal/mlxthread`'s own
`thread_affinity_test.go` / `TestDoUsesSameOSThread`.

## THE DECISIONS

### D1 — whose gemma4 MLX vision survives

Upstream's `PrepareMedia(segments)` has **no per-request budget**: it reads a
fixed per-checkpoint soft-token budget (`visionSoftTokenBudget()` ∈ {70, 140,
280, 560, 1120}, anything else rejected at load) and resizes with its own
`visionTargetSize`. Ours (`PrepareMediaWithBudget`, `resolveImageBudget`,
ADR 0008's `BudgetFillSize` ladder, ADR 0021's seam) is what makes
`image_min_tokens` / `image_max_tokens` work on the MLX path. The two towers
share symbol names with incompatible definitions and different bidirectional
mask designs (`buildMasks` + `use_bidirectional_attention` upstream vs our
`visionChunkMask` in `Attention.Forward`).

| option | cost | risk |
|---|---|---|
| **A. Keep ours for this fold; no gemma4 audio yet.** Take upstream's `gemma4.go` changes only where they do not touch vision/audio wiring; keep our four vision files; leave `x/mlxrunner/model/audio` unused; restore `isGemma4Renderer`. | The conflict recurs at v0.33.4; no audio. | **Lowest.** Same vision code as deployed; goldens, flat/aspect ladders and the budget knob all remain valid instruments — the MLX pin bump is the only thing they test. |
| **B. Adopt upstream, re-graft the budget seam.** Supersede ADR 0021 by ADR; implement `MediaBudgetModel` on upstream's model (map `image_min/max_tokens` onto the fixed budget set, or onto `maxPatches` in `visionTargetSize`); re-baseline goldens and `mlx-metal` ladders (`aspect_ladder` values 1066/1058/1091 *will* move — upstream floors sides differently). | 2–3 days + a full MLX re-baseline; ADR 0003/0007/0008 policy re-stated on upstream's resize. | Medium: a measured change to the fork's oldest vision claim. Must be a spike with numbers first. |
| C. Adopt upstream as-is. | none | **Rejected**: `image_max_tokens` becomes a silent no-op on MLX gemma4 (the type assertion at `x/mlxrunner/media.go:291` fails quietly) — exactly what ADR 0021 decision 3 forbids. |

**Recommendation: A for this fold, B as its own task.** A folds everything
else to 0.33.3 without touching a measured surface; B is decided on a spike
branch that builds upstream's vision, runs `TestVisionGoldenParity` and the
`mlx-metal` ladders, and reports the delta — the same "decision, not merge
resolution" discipline as ADR 0033.

### D2 — whose two-pass structured-output metrics fold wins

Upstream (`855f4bf9`) adds `IncludeIntermediateMetrics` + `firstPassMetrics`
and folds the restart as "retain the original prompt metrics, fold in the
second prefill". ADR 0010 derives the prompt count from pass two's
cache-inclusive prefill minus the pure-text continuation delta, because image
embedding tokens are invisible to textual counting. They give **different
numbers on vision requests.**

**Recommendation: keep ADR 0010's derivation, layered on upstream's plumbing**
(the signal and the arithmetic are compatible). Adopting upstream's fold
instead supersedes ADR 0010 and re-records every `think + format` cell.

## Pre-fold housekeeping (shrinks the conflict surface; each its own PR)

1. **Merge #211** — `x/mlxrunner/client.go:578`, `unwind.go:40`,
   `client_env_test.go:22` cite `docs/maxusai/mlx-thrash-check-masks-as-cudagraph.md`,
   which exists only there. Add the one-line status ("implemented in #212").
2. **Delete `x/mlxrunner/constrain.go`, its `speculate.go` call sites and the
   `constrain*_test.go` files** (ADR 0033's stated follow-up; `attachGrammar`
   has one match — its own definition). Removes one of the 18 `mlxtest.Setup`
   ports before it has to be made.
3. **#201** merge with a *superseded by ADR 0033* header (it is the ADR's
   evidence); **#213** amend per #254/#257/#258 (12b descoped; throughput
   caveat; n≥3 repeat via targeted re-run) then merge; **#210** merge or close.
4. **Fix `release_matrix.py:35`** — "Output quality" keys on `{"text_baseline",
   "quality"}` but `checks.py:942` emits `extraction_quality`, so the column
   is green regardless. The fold's green matrix is judged on it.
5. **Tag hygiene (ADR 0032):** deployed `:11497` is `0.33.2-dynres-5-g2b95b4a`
   = the #238 merge, five first-parent commits past `v0.33.2-dynres`. Either
   cut `v0.33.2-dynres.1` at `2b95b4a5` or record the equivalence; the README
   says the tag is the rollback point and today it is not what is running.

## Fold sequence and gates

Branch `task/upstream-sync-0.33.3` (this one). `main` is untouched until
Gate 4 is green. Rollback point throughout: `v0.33.2-dynres` lineage and the
running `maxusai/ollama:sync-0.33.2` container — do not delete that image.

**Gate 0 — D1 and D2 recorded** (ADR if D1-B or D2-upstream). Then the
housekeeping PRs above are merged, and this branch is rebased on them.

> **This branch is 9 commits behind `origin/main`** as of the merge (base
> `b54d4d0d`; main is at `64069bf6` with #266, #213 and #268 landed since).
> Those nine touch six files — `README.md`,
> `docs/maxusai/vision-campaign-2026-08-24-sync15nt-thinkon.md`,
> `vision-learnings-log.md`, `vision-model-recommendations.md`,
> `preflight/release_matrix.py`, `preflight/test_verdicts.py` — and **none of
> them is touched by this merge** (verified by set intersection), so bringing
> main in is conflict-free. Gate 2's numbers were measured without them: the
> control run of `origin/main` executes 100 `test_verdicts.py` tests to this
> branch's 94, the difference being #266's six new quality-column tests.

**Gate 1 — mechanical merge, no build.** `git merge v0.33.3`; resolve in this
order: `mlxtest.go` (theirs + port 18 sites) → `mlx/{mlx,stream}.go` (theirs;
`memory.go` off `mlxCall`; `ClaimOSThread` callers onto `mlxthread`) →
`x/models/gemma4/*` per D1, whole files → `server/{images,model_list_cache}.go`
+ `isGemma4Renderer` per D1 → `llm/llama_server.go`, `server/routes.go`,
`x/mlxrunner/{pipeline,client}.go` per D2 → `cmake/mlx` union → workflows.
Exit check: `git grep -n 'mlxCall(\|ClaimOSThread()\|mlxtest\.Setup\|isGemma4Renderer'`
returns only what D1 intends.

Resolution recipes, read from v0.33.3 so Gate 1 is mechanical:

- **`ClaimOSThread` → `mlxthread`.** Upstream's runner already pins the MLX
  thread with `mlxthread.Start("mlxrunner", init)` (`x/mlxrunner/server.go`)
  and `x/create/mlxthread.go` uses `runtime.LockOSThread()` for the process
  lifetime — that *is* ADR 0017's guarantee, expressed upstream's way. Drop the
  fork's `ClaimOSThread` and its `__thread _mlx_thread_owned` flag; keep the
  invariant text in AGENTS.md/ADR 0017 pointing at `mlxthread`.
- **`memory.go`.** Upstream has no `MemoryLimit` / `SetMemoryLimit` /
  `SetCacheLimit` — the fork's VRAM/cache ceiling (`OLLAMA_MLX_MEMORY_LIMIT`,
  `budgetWithOverride`) depends on them. Re-express the three on
  `mlxError[T](v T) error` / `lastError()`; confirm `mlx_set_cache_limit`,
  `mlx_set_memory_limit`, `mlx_get_memory_limit` exist in the regenerated
  MLX-C bindings before assuming the port is a rename.
- **`mlxtest`.** New API: `mlxtest.Run(t, func(*mlxtest.T))`,
  `mlxtest.RunSubtest(t, name, fn)`, `mlxtest.SkipIfUnavailable(t)`; the
  callback runs on the package's shared MLX thread (`x/internal/mlxthreadtest`).
  Port the 18 `Setup` sites by wrapping the body; AGENTS.md's "build arrays
  inside the subtest" rule maps onto `RunSubtest`.
- **D2 union.** Upstream adds `IncludeIntermediateMetrics` to the runner
  request, sets `includeIntermediateMetrics := req.Format != nil && currentFormat == nil`
  in `ChatHandler`, captures `firstPassMetrics`, then folds with
  `PromptEvalCount = firstPassMetrics.PromptEvalCount; EvalCount += …;
  EvalDuration += … + r.PromptEvalDuration`. Keep all of that wiring and the
  new `PromptEvalCachedCount` pass-through; replace only the
  `PromptEvalCount` line with ADR 0010's `transitionPassMetrics()` derivation
  (pass two's cache-inclusive prefill minus `transitionPromptDelta`), and keep
  #238's `deferring` gate on both transition sites. `TestChatNullFormatIsNotConstraining`
  and the restart tests (`TestChatWithPromptEndingInThinkTag/…`) are the
  semantic gate; add one asserting the folded count on a vision request.

**Gate 2 — no-GPU tests** (golang:1.26 container, `-u 1000:1000`, caches on
the 8 TB array):
```sh
go build ./... && go vet ./...
go test ./llm/ -run 'TestImageTokensForSize|TestKVCacheType'
go test ./server/ ./model/... ./llm/ ./api/ ./convert/ ./x/structured/ ./x/mlxrunner/... ./x/internal/...
python3 docs/maxusai/vision-suite/preflight/test_verdicts.py
```
MLX-tagged tests skip without a payload — a skip is not a pass.

**Gate 3 — CUDA image on the `bigdisk` builder** (`scripts/build_docker.sh`
with the GOFLAGS version stamp; state on the array; never pipe build output
to the scratchpad; expect the ~3 h MLX nvcc cache miss). Proofs: the six
compat patches applied during FetchContent (`apply-git-patches.cmake` fatals
otherwise); `libquadmath.so.0` + `libcusolver`/`libcusparse`/`libnvJitLink`
in `lib/ollama`; the new `ollama-go-license` step succeeded;
`grep -c -- --image-max-tokens /usr/bin/ollama` = 1.

**Gate 4 — preflight, re-measure first, re-pin second** (the #239 precedent:
`payload_pin` is expected to be the sole failure before the identity moves,
and the ladders came back byte-identical at the last two pin bumps —
measure anyway):
```sh
python3 docs/maxusai/vision-suite/preflight/measure_ladder.py --host http://127.0.0.1:<port> \
  --profile cuda-dynres-903 --arch {nemotron_h_omni,gemma4,qwen35} --container <name> --stride <n>
# paste the emitted [expect.…] blocks whole (ADR 0012 rule 8); then set
# [profile.cuda-dynres-903].llama_cpp_build = "0f3a71be1"
setsid nohup python3 docs/maxusai/vision-suite/preflight/preflight.py --host http://127.0.0.1:<port> \
  --platform cuda --image-tag maxusai/ollama:sync-0.33.3 --quality --out runs/full-0333.json > runs/full-0333.log 2>&1 < /dev/null &
```
`poison_probe` with the `801` meter on; exit 0 required (3 = re-run alone;
4 = a baseline step was skipped; 2 = wrong port or missing profile — never
`--allow-unmeasured` past it). Metal host: new `[profile.mlx-metal-0-33-3]`
(`version_pattern '^0\.33\.3-maxusai-…'`, `mlx_build = "37c26e5755da…"`),
seeded from `0-33-2` then re-measured incl. `aspect_ladder`;
`OLLAMA_VISION_E2E=1 go test ./x/mlxrunner/ -run TestVisionGoldenParity`
against the four goldens (the MLX pin moved fused-kernel rounding once
before — #225).

**Gate 5 — vision-suite spot-check** (`run_engine_compare.sh` only,
`OLLAMA_MAX_LOADED_MODELS=1`, new `TAG_PREFIX`; ADR 0032: `v0.33.3-dynres` is
a new build identity, never folded into a `0.33.2` table): the five nvfp4
think-off cells. **Baseline on the CUDA host is the last CUDA MLX think-off
data — `sync15_1_` (12b/26b/31b/qwen3.8) and `sync15nt_1_` (qwen3.6), build
`0.32.14-dynres-108-g76918a7`** — because the mlx0330nv/mlx0332nv campaigns
(#229/#257, "reproduced to three decimals") are Metal-host measurements. Pass
criterion: converged arms identical on the score columns within run-to-run
noise (n = 1, #258), no arm newly capped or errored. The Metal half of this
gate is Glenn's, against mlx0332nv1. Under D1-B add the gemma4 MLX cells at
n≥5 with a positive control. Re-run the qwen3.5-MoE MMQ padding gate at
b10760 and the qwen2.5vl poison probe (both `cuda-dynres-903`).

**Gate 6 — tag, release, deploy.** `v0.33.3-dynres` on the merge commit,
`v0.33.3-maxusai` lineage on the Mac; GitHub Release with the generated green
matrix (`release_matrix.py --version 0.33.3-dynres runs/full-0333.json`, full
run only — a later `--skip-pinned` smoke would overwrite green with
*skipped*); README **Current fold** pointer; deploy as
`ollama-0.33.3-dynres-<n>-g<sha>` built from the tag; `:11434` stays parked.

## Metal half — handoff (Glenn's host, 10.8.0.3; SSH refused from the CUDA host)

Once #264 passes the CUDA gates, the Apple side is the same three steps as
#243/#225, on the fold branch's tree:

1. Native build of the branch (`0.33.3-maxusai-<sha>` stamp), served on
   `:11437` for preflight (`:11436`/`serve-apple-mlx.sh` for the campaign);
   confirm the runner logs `"MLX version"=37c26e5755da…` (the payload moved,
   so `mlx_payload_pin` must see the new pin, not `c793734e`).
2. Cut `[profile.mlx-metal-0-33-3]` in `expectations.toml`: `version_pattern
   = '^0\.33\.3-maxusai-[0-9a-f]{7,40}$'`, `mlx_build = "37c26e5755da637255d57ea34b4879196a485301"`,
   `[expect.…]` blocks seeded from `mlx-metal-0-33-2` and **re-measured**
   (`measure_ladder.py`, incl. `aspect_ladder`); never widen `0-33-2`'s
   pattern (ADR 0011 rule 5). Then the full `--platform mlx-metal` preflight,
   detached, exit 0 required.
3. `OLLAMA_VISION_E2E=1 go test ./x/mlxrunner/ -run 'TestVisionGoldenParity|TestVisionEndToEnd'`
   against the four goldens — the MLX pin moved fused-kernel rounding once
   before (#225: bound raised with a `gemma4:26b-mlx-bf16` control); a
   recalibration needs the same control, not a bound edit.
4. Campaign spot-check `mlx0333nv1_`, think-off, five nvfp4 tags, compared
   to `mlx0332nv1` (#257) — the three-decimal criterion applies **here**,
   same host, same suite.

Artifacts back into the tree: the preflight run JSON under `preflight/runs/`
(the green matrix regenerates from it), the profile, and the campaign doc.

## Do not touch / defer

- `903` — keep; functional gate before any pre-Ada GPU deploy.
- `GGML_CUDA_CUBLAS_COMPUTE_TYPE` / `applyArchServerEnvs` — do not simplify.
- `quadmath` in `MLX_INCLUDE_REGEXES` — the symptom appears far from the cause.
- `801` — the instrument behind `poison_probe`'s node evidence.
- `x/mlxrunner/prefix_cache.go` `mediaRestoreFloor` — untouched upstream; if
  D1-B, re-verify it against upstream's `buildMasks` (no `SeqOffsets[0]==0`
  assumption there).
- gemma4 audio — under D1-A not shipped; under D1-B ships unmeasured unless a
  probe is added; say which.
- Never widen `mlx-metal-0-33-2`'s pattern to 0.33.3.
- ROCm stays gated at the 0.32.1 base (`amd-upgrade-gate.md`); `mlx-cuda` and
  `cpu` remain unmeasured surfaces and must read *not run* in the matrix.

## Acceptance criteria

1. ☑ D1 and D2 decided and recorded (2026-09-04, above). No ADR changes under D1-A / D2-as-decided; the D1-B spike carries its own ADR if adopted.
2. ☐ Housekeeping: ☑ #211, ☑ #201, ☑ #265 (point-tag patterns + ADR 0032 amendment), ☑ `v0.33.2-dynres.1` cut at `2b95b4a5` (2026-09-04), ☑ #213 amended (`adab78e4`), ☑ `release_matrix.py` quality column (#266, `1b0c716b`); ☑ `constrain.go` deletion (#267, merged 2026-09-04 with the #269 lint fix folded in; `main` merged back into this branch at `ddb675a3`, zero conflicts, and main's test workflow is green again); ☐ #210 (merge or close). **The three ☑ landed on `main` after this branch was cut — see the note under Fold sequence.**
3. ☑ Merged 2026-09-04 on `task/upstream-sync-0.33.3` (`git merge --no-ff
   v0.33.3`, second parent `b79067b0`). All 28 conflicts resolved per the
   table above; zero conflict markers left in the tree. Gate 1 exit grep
   (`mlxCall(\|ClaimOSThread()\|mlxtest\.Setup\|isGemma4Renderer`) returns
   only `isGemma4Renderer` in code — its definition in
   `server/renderer_resolution.go` and its two uses in `server/images.go` and
   `server/model_list_cache.go` — plus prose in ADR 0017, ADR 0021 and this
   doc. `git diff --name-status v0.33.3 HEAD -- x/models/gemma4` is the D1-A
   shape: `D` on `audio.go`, `audio_test.go`, `process_image.go`; `M` on
   `gemma4.go`, `media.go`, `media_test.go`, `vision.go`, `vision_test.go`;
   `assistant.go`, `gemma4_test.go`, `gemma4_moe_test.go` identical to upstream;
   `process_audio.go`/`process_audio_test.go` excluded after Gate 2 (lint). Payload pins are
   upstream's: `b10760` / `37c26e5755da…` / `c74db5307cc8…`.
4. ☑ Gate 2 green 2026-09-04 — and re-run after `main` was merged back (`ddb675a3`) and
   `process_audio*.go` excluded (`ec7c35a5`): `go build ./...`, full `go test ./...` = 67
   packages ok, `test_verdicts.py` 100 OK, **golangci-lint 0 issues at repo root**. GitHub
   CI on `ec7c35a5` (#264): all jobs green — `test` ×3 OS, upstream's new `race` ×2 and
   `go_license`, `patches` ×2 and `patches-ggml` (the compat band applied against b10760
   in CI), `go_mod_tidy`, `changes`. Original Gate 2 record follows.
   Gate 2 (first run) 2026-09-04 (`golang:1.26.0`, `-u 1000:1000`, caches on the
   8 TB array). `go build ./...` clean. `go test ./llm/ -run
   'TestImageTokensForSize|TestKVCacheType'` clean. The full set
   (`./server/ ./model/... ./llm/ ./api/ ./convert/ ./x/structured/...
   ./x/mlxrunner/... ./x/internal/... ./x/models/... ./x/create/...`):
   **4230 pass, 0 fail, 237 skip**. `test_verdicts.py`: 94 tests OK, 6
   skipped (quality arm — pre-existing, `summarize_engine_compare` import).
   Two caveats, both pre-existing and both verified against a control run of
   `origin/main` in the same image:
   - `go vet ./...` reports exactly one finding,
     `tokenizer/bytepairencoding_test.go:542:5: result of slices.Collect call
     not used`. The file is byte-identical to v0.33.3's and unchanged since
     before the merge base; the control reports the same single finding. No
     new vet findings.
   - `gofumpt -l` over the 120 Go files this merge touched lists one,
     `server/routes_generate_test.go`; its two hunks are pre-existing fork
     code in `TestChatFormatPassthrough`'s table (identical hunks on
     `origin/main`), outside anything this merge changed, and were left
     rather than adding unrelated reformatting to a merge commit.

   **A skip is not a pass.** 188 of the 237 skips are "MLX not available:
   failed to load MLX dynamic library" — the golang image carries no native
   payload — so every MLX kernel path is unexercised, including
   `TestVisionGoldenParity` and `TestVisionEndToEnd` (which also gate on
   `OLLAMA_VISION_E2E`). Both compile and skip cleanly. That coverage is
   Gate 4's.
5. ☑ Gate 3 (2026-09-04): `maxusai/ollama:sync-0.33.3` built on `bigdisk` from `242cad55`
   (cache-warming build of `eaef4905` 4 h 40 m, then the stamped rebuild 8 m 53 s), stamp
   `0.33.2-dynres.1-39-g242cad5` (pre-tag; ADR 0032 equivalence to be recorded when
   `v0.33.3-dynres` is cut on the merge commit). Proofs: build rc 0 with all six compat
   patches applied by `apply-git-patches.cmake` (the compat sources compiled into `llama`
   and `mtmd`; a failed patch fatals the configure); `lib/ollama/mlx_cuda_v13` carries
   `libquadmath` plus upstream's new `libcusolver`/`libcusparse`/`libnvJitLink`, and `ldd`
   resolves every dependency of `libmlx.so`/`libmlxc.so` inside the image (0 unresolved);
   the `ollama-go-license` step produced `GO_LICENSE`; `--image-max-tokens` is in the
   binary (go_patch_marker). Log: `claude-scratch/build-0333-final.log` on the 8 TB array.
6. ◐ Gate 4: ☑ CUDA ladders re-measured on the `:11437` canary (`sync-0.33.3`, 2026-09-04
   17:31–17:41) **before** the pin moved — nemotron `[266, 266, 578, 2306, 3270]`, gemma4
   `[1102 ×5]`, qwen35 `[1034, 1034, 1034, 2306, 4082]`, byte-identical to the b10630 rows
   (third pin bump in a row with unchanged ladders); ☑ `payload_pin` moved to `0f3a71be1`,
   no expected value edited (the #239 procedure); ☑ full preflight (2026-09-04 17:46–17:52, `:11437` canary, `--quality`): **VERDICT PASS —
   20 PASS / 8 SKIP / 0 FAIL**, artifact `preflight/runs/full-0333.json` (on host, copied to
   the 8 TB array). `version` matched `cuda-dynres-903` via #265's widened pattern;
   `payload_pin` matched `0f3a71be1`; `poison_probe` decoded healthily with the `801` meter
   clean over 33 nodes (the #215 gate holds on b10760); ladders/pinned budgets/think_format
   PASS on all three arches. SKIPs are recorded gaps, not failures: no `aspect_ladder` or
   quality thresholds on this profile (×3 each), qwen35 pinned-budget not applicable,
   `mlx_payload_pin` unpinned on `mlx-cuda`. Same PASS count as the 0.33.2 fold's full run.
   Note the run is stamped `0.33.2-dynres.1-39-g242cad5` (pre-tag); Gate 6 rebuilds from
   the tagged merge commit and re-runs the full preflight so the green matrix carries the
   `0.33.3-dynres` identity without an equivalence argument; full preflight exit 0 with `poison_probe` corroborated; `mlx-metal-0-33-3` profile cut and PASS; goldens PASS (or recalibrated with a bf16 control, as #225).
7. ☑ Gate 5 (CUDA, 2026-09-04 17:53–19:27, `vsuite-0333` on `:11503`, `mlx0333cu_1_` think=false,
   `num_ctx` 8192, one runner, cold restart per cell; scores copied to
   `preflight-runs/scores-0333/`). Against the sync15 CUDA baselines, converged arms compared
   on every score column:
   - gemma4:12b 27/27 (20 identical, rest ±0.004; `bbox_contract_adv_norm1` 0/6 → 6/6)
   - gemma4:26b 26/27 (14 identical, rest ±0.005; `bbox_contract_real_1img` 1 → 0 hits at
     IoU 0.15 → 0.05 — poor in both runs)
   - gemma4:31b 24/27 (14 identical, rest ±0.005; `bboxm_free_noanc_pos` 0.925 → 0.960)
   - qwen3.8:27b 27/27 (11 identical, rest ±0.01 both ways)
   - qwen3.6:35b-a3b 27/27 (8 identical; `bbox_contract_reasoning` 0 → 6/6 and
     `bboxm_free_anc_named` 1 → 6/6 up; `document_single` name_bbox 0.646 → 0.504 down;
     invoice fields identical)
   No systematic direction; the movements are the temperature-0 run-to-run class #258
   documents. **The four non-converged arms are all `cudaMallocAsync … out of memory`** on
   26b (`multi_3img_anchored`) and 31b (`multi_3img`, `bbox_contract_perobject`,
   `bbox_contract_adv_norm1`) — two of them small single-image requests, so not request
   size: the runner's MLX ceiling is derived from free memory at cell start (61.7 GiB here)
   while the production endpoint on the same GPU (`:11497`, gemma4:31b q4 resident, a client
   sending `/api/chat` every 30–80 s throughout) moved underneath it. The fold's machinery
   behaved: `guardClose` reported the OOM as the cause, the scheduler expired the model, the
   next request reloaded. The memory-limit port (`mlxError` for `mlxCall`) is faithful and
   the limit is logged as applied on every runner start; the cache limit is env-driven and
   unchanged. ☑ Discriminator (19:28–20:01, `vsuite-0333-capped` on `:11504`, `mlx0333cuRR_`, the runner
   logging `MLX memory limit overridden from the environment: requested 40 GiB, derived
   61.8 GiB, using 40.0 GiB`): the same four arms 3× back to back — **12/12 converged, 0
   OOMs**, every metric equal to the sync15 baseline (31b `bbox_contract_perobject` 6/6 at IoU
   0.923 vs 0.851; `adv_norm1` 0/6 in both). The OOMs were shared-GPU contention, not the
   fold. Serving note (corrected 2026-09-05, see 7b): on this host give a campaign or canary
   container **headroom with `OLLAMA_GPU_OVERHEAD`** — the runner subtracts it from free VRAM
   when it sizes itself, so the ceiling follows what is actually free; a hard
   `OLLAMA_MLX_MEMORY_LIMIT` is an absolute bound and, set near the workload's peak, it stalls
   instead of OOMing (vision-suite README, #277). ☐ qwen3.5-MoE MMQ padding gate at b10760 —
   source-inspected only (the b10760 change widens MMQ to MoE on pre-DP4A cards; Blackwell is
   unaffected). ☑ qwen2.5vl poison probe re-run at b10760: PASS inside the Gate 4 preflight.
   **Gate 5 (CUDA) result: no regression.**
7b. ☑ **Gate 5 completed on CUDA, 2026-09-05** — the think-on half, and the same-platform
   0.33.2 baseline that Gate 5 had lacked (its comparator above is the sync15 0.32.14 build;
   the 0.33.2 vision campaigns were Metal-host measurements and never mix with CUDA ones).
   - **Think-on** (`vsuite-0333on` on `:11505`, `mlx0333cu_1_*_thinkon`, ladder
     16384→32768→65536) against `sync15nt_1_`: 12b DESCOPED by policy; 26b **25/27 = 25/27**
     (same two arms standing); 31b **27/27 = 27/27** at rung 1 (17 identical, rest ±0.003);
     qwen3.8 **27/27 = 27/27** at rung 1; qwen3.6 **24/27 = 24/27** at the ceiling with two
     of the three standing arms the same (`scene_single`, `bbox_contract_real_1img`; the third
     `scene_single_pinned` now vs `bboxm_free_anc_pos` then). Per-arm scores on qwen3.6 swing
     both ways run to run (4/23 comparable arms identical, 0→5 and 6→4 hits in the same cell);
     the converged count is the stable metric. **No regression.**
   - **What the think-on run also found.** The container had been started with a hard
     `OLLAMA_MLX_MEMORY_LIMIT` of 40 GiB (the discriminator's value). At rung 32768 the
     request before the trouble peaked at 39.98 GiB; the next request got its 371-token prompt
     processed and then the runner logged nothing for 30 min, and the three requests behind it
     never reached prompt processing — each burned its own 1800 s client timeout. Two hours
     lost to one stall, cleared only by the between-rung restart. At the ceiling MLX does not
     OOM, it crawls: the cache-thrashing check that would abort it is off by default (#212).
     Glenn's call: the container was replaced mid-rung-3 with an identical one at 56 GiB (the
     client's transport retry carried the arm in flight); the four affected arms all completed
     at rung 3, so their recorded rung of convergence is higher than baseline for three of them
     — an artefact of the stall, not the model. Evidence: `preflight-runs/vsuite-0333on-runner.log`.
     **Lesson: the runner already has a headroom knob, `OLLAMA_GPU_OVERHEAD`; use it, not a cap.**
   - **0.33.2 think-off baseline on CUDA** (`vsuite-0332cu` on `:11506`, `maxusai/ollama:sync-0.33.2`,
     `mlx0332cu_1_*_thinkfalse`, `OLLAMA_GPU_OVERHEAD` 16 GiB → ceiling 45.8 GiB, **0 OOMs**),
     against the 0.33.3 cells of criterion 7, converged arms compared on every score column:

     | cell (think-off, CUDA) | 0.33.2 | 0.33.3 | identical / comparable | rest |
     |---|---|---|---|---|
     | gemma4:12b-nvfp4 | 27/27 | 27/27 | 24 / 27 | ±0.003 |
     | gemma4:26b-nvfp4 | 27/27 | 26/27 (1 OOM) | 11 / 26 | ±0.005; `positional_1img` 0.971→0.859, `real_1img` 1→0 hits (unstable arms) |
     | gemma4:31b-nvfp4 | 27/27 | 24/27 (3 OOM) | 13 / 24 | ±0.004 |
     | qwen3.8:27b-nvfp4 | 27/27 | 27/27 | **27 / 27** | — |
     | qwen3.6:35b-a3b-nvfp4 | 27/27 | 27/27 | 10 / 27 | ±0.01; 0.33.3 higher on `bbox_contract_reasoning` 0→6 and `_multi` 0→3 |

     Every arm 0.33.3 lacks is one of the four OOM'd arms, all of which re-ran clean.
   - **Memory, per-request `peak memory` from the runner** (same models, same arms):

     | | n | p50 | p90 | max |
     |---|---|---|---|---|
     | 0.33.2, gemma4 cells | 83 | 25.8 GiB | 34.2 GiB | 37.1 GiB |
     | 0.33.3, recorded sample (the three gemma4 cells) | 84 | 24.4 GiB | 33.1 GiB | 36.3 GiB |
     | 0.33.2, whole run | 144 | 26.6 GiB | 36.6 GiB | 41.1 GiB |

     Per model on 0.33.2: 12b 24.8, 26b 34.7, 31b 37.1, qwen3.8 39.2, qwen3.6 **41.1 GiB** — the
     last above the 40 GiB the discriminator used, which happened to touch only 26b/31b arms.
     Like for like the builds are within ~1 GiB at every quantile, 0.33.2 the higher. **0.33.3
     does not allocate more for the same work; the Sep-4 OOMs were zero-headroom contention.**
     Data: `preflight-runs/peakmem-0332cu.log`, `scores_mlx0332cu_1_*`.
   - Follow-ups landed from this: #276 (MLX admission prices the context rung, headroom
     calibrated on GPU), #277 (driver evicts on exit; README advice corrected to headroom),
     #279 (a `num_ctx` change reloads an MLX runner, either direction), #280 (comparing
     builds goes through the ADR 0012 generators; errored arms render `error`).
7c. ☑ **`main` after those merges validated on CUDA, 2026-09-06** (Glenn's order: the OOM
   cases, then the think-off campaign, then preflight + the clamp on a real load), on
   `maxusai/ollama:main-a523d60b` — the tagged image's payload with `main`'s Go binary,
   payload byte-identical. Full write-up with generator tables only:
   [vision-campaign-2026-09-06-main-admission-validation.md](../vision-campaign-2026-09-06-main-admission-validation.md).
   - The four Sep-4 OOM arms ×3 on 26b/31b: 24/24 converged, 0 OOMs, scores equal to the
     Sep-4 discriminator; under a 35 GiB cap the rung that cannot fit (31b at 65536, need
     37.6 GiB) is **refused at admission in 4 s** instead of aborting mid-prefill.
   - Think-off, five models: 135/135 arms converged at 8192, 0 OOMs; by T2 every quality
     row equals the tagged 0.33.3 cells and the 0.33.2 baseline (0.33.3's own Sep-4 OOM
     cells render `error`); contract matrices identical to 0.33.2, and to 0.33.3 bar its
     two OOM'd arms and one single-arm flip on qwen3.6 under an n=3 repeat.
   - Preflight `cuda-dynres-903` on the `main` binary: PASS 20 / SKIP 8, 28 checks identical
     to the tagged run. The automatic-context clamp fired on a real load (31b 262144 → 65536,
     served; 12b not clamped).
   - #279 live proof, FIX vs CONTROL: a rung change re-admits (larger refused when it cannot
     fit, smaller releases the window, same rung served warm); `main` before it never reloaded.
   **Deploy note:** the tagged `sync-0.33.3` remains the validated deployable exactly as
   before; a build from `main` is now validated on this host to the same depth on the CUDA
   think-off surface plus the admission behaviour, and would need the Metal half like the tag.
8. ☑ Landed 2026-09-04 (Glenn: go for steps 1–3, deploy held): #264 merged as `0c4f09d4`;
   annotated `v0.33.3-dynres` on that commit (`git describe` → `v0.33.3-dynres-0-g0c4f09d`);
   image rebuilt from the tag (`maxusai/ollama:sync-0.33.3`, stamp `0.33.3-dynres-0-g0c4f09d`,
   11 m 30 s on the warm cache); **full preflight on the tagged build: VERDICT PASS 20 / 8 / 0**
   (`preflight/runs/full-0333-tagged.json`, same result as the pre-tag run); green matrix
   generated from that run (cuda row green, Output quality *skipped* — no thresholds for
   this profile's arches); GitHub Release
   [`v0.33.3-dynres`](https://github.com/MaxusAI/ollama/releases/tag/v0.33.3-dynres) with the
   fork's notes and the matrix; README pointer + matrix in #273. `-maxusai` lineage tag is the
   Metal half's (handoff above).
8b. ☑ **D1-A verified end to end (2026-09-04 23:12).** The decision this fold turns on —
   the per-request image budget survives on MLX gemma4 — had no probe: preflight's
   `pinned_image_token_budget` covers the llama-server path only, and Gate 5 ran default
   budgets. `vision-suite/probe_mlx_image_budget.py` sweeps `image_max_tokens` over gemma4's
   soft-token ladder against one image and reads `prompt_eval_count` back. Both builds, same
   image (`ladderimgs/2048x1152.png`), `gemma4:12b-nvfp4` (MLX):

   | `image_max_tokens` | default | 1120 | 560 | 280 | 140 | 70 |
   |---|---|---|---|---|---|---|
   | `sync-0.33.3` (`0.33.3-dynres-0-g0c4f09d`) | 1122 | 1122 | 549 | 286 | 142 | 88 |
   | `sync-0.33.2` (`0.33.2-dynres-5-g2b95b4a`) | 1122 | 1122 | 549 | 286 | 142 | 88 |

   Five distinct readings on both — the knob is live, not silently ignored — and **byte-identical
   between the two builds**, so D1-A preserved the seam exactly. Corroboration across engines: at
   the 560 rung the MLX prompt is 549 tokens against a ~20-token text prefix, i.e. ~529 image
   tokens, matching preflight's independently measured `pinned 560 -> 529` on the llama-server
   path. Default equals 1120, as ADR 0008 requires. 0 server errors; the probe capped the MLX
   pool per #272 and tore its container down in a trap.

9. ☐ **Deploy held by Glenn.** `:11497` still serves `0.33.2-dynres-5-g2b95b4a`
   (`v0.33.2-dynres.1`); when deploying: `ollama-0.33.3-dynres-0-g0c4f09d` from
   `maxusai/ollama:sync-0.33.3`, `sync-0.33.2` retained as rollback, README **Deployed** line
   updated (ADR 0032 amendment), and — on this shared GPU — consider `OLLAMA_GPU_OVERHEAD` (headroom; not a hard cap, see 7b)
   for any campaign/canary container alongside it (#272).

## Effort

Housekeeping ½ day · merge + ports 1–2 days · image ~3 h · CUDA preflight
~1 h (+quality) · Metal profile + goldens ½ day · Gate 5 spot-check ~4 h GPU.
D1-B adds a 2–3 day spike before its own ADR.
