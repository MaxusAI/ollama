# TASK: fold upstream v0.33.3 into main — plan

**Opened:** 2026-09-04. **Status:** PLAN, no merge attempted on `main`. Two
decisions (D1, D2) are Glenn's before the merge starts; everything else below
is mechanical and verified. Assessment by two max-effort reviews (PR history
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

| file(s) | resolution |
|---|---|
| `x/models/gemma4/{media,vision,media_test,vision_test}.go` (add/add), `gemma4.go`, `gemma4_test.go`, `gemma4_moe_test.go` | **D1**, whole-file per side; never hand-merged |
| `server/routes.go` (old 2747–2801), `x/mlxrunner/pipeline.go`, `x/mlxrunner/client.go`, `llm/llama_server.go` | **D2**; preserve `guardClose` call sites, the `stopper` block, `applyCompletionFormat`, `visionServerArgs`, `kvCacheFlagValues`, `ggmlCublasComputeTypeEnv`, `mlxRunnerEnvDefaults` |
| `x/mlxrunner/mlx/{mlx,stream}.go`, `mlx/thread_test.go` | take theirs; re-express ADR 0017's guarantee on `x/internal/mlxthread` (upstream's own answer); port `memory.go` off `mlxCall` onto `mlxError`; confirm `mlx_set_cache_limit` / `mlx_set_memory_limit` / `mlx_get_memory_limit` survive the MLX-C regeneration |
| `x/internal/mlxtest/mlxtest.go`, `x/mlxrunner/cache/*_test.go`, `model/embedding_test.go`, `sample/sample_test.go`, `x/models/{laguna,qwen3_5}/*_test.go`, `x/mlxrunner/client_test.go` (add/add) | take theirs; port the 18 `Setup` sites to `Run`/`RunSubtest`; concatenate the two `client_test.go` |
| `server/images.go`, `server/images_test.go`, `server/model_list_cache.go` | take upstream's deletion of the gemma4 capability suppression; resolve `isGemma4Renderer` consistently with D1 (restore it under D1-A; drop the branch under D1-B) |
| `cmake/mlx/CMakeLists.txt` | union: keep our `$ORIGIN` RPATH block and **`quadmath`**, take `cusolver cusparse nv[Jj]it[Ll]ink`, `OLLAMA_LIB_DIR`, license installs. Losing `quadmath` = a CUDA MLX payload that fails `CheckInit()` |
| `.github/workflows/test.yaml` | union of path filters (as #232) + upstream's MLX unit-test job |
| `docs/api.md` | take theirs, re-add our option docs |

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

**Gate 1 — mechanical merge, no build.** `git merge v0.33.3`; resolve in this
order: `mlxtest.go` (theirs + port 18 sites) → `mlx/{mlx,stream}.go` (theirs;
`memory.go` off `mlxCall`; `ClaimOSThread` callers onto `mlxthread`) →
`x/models/gemma4/*` per D1, whole files → `server/{images,model_list_cache}.go`
+ `isGemma4Renderer` per D1 → `llm/llama_server.go`, `server/routes.go`,
`x/mlxrunner/{pipeline,client}.go` per D2 → `cmake/mlx` union → workflows.
Exit check: `git grep -n 'mlxCall(\|ClaimOSThread()\|mlxtest\.Setup\|isGemma4Renderer'`
returns only what D1 intends.

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
think-off cells — on 0.33.2 they reproduced 0.33.0 to three decimals (#257);
the same is the pass criterion here. Under D1-B add the gemma4 MLX cells at
n≥5 with a positive control. Re-run the qwen3.5-MoE MMQ padding gate at
b10760 and the qwen2.5vl poison probe (both `cuda-dynres-903`).

**Gate 6 — tag, release, deploy.** `v0.33.3-dynres` on the merge commit,
`v0.33.3-maxusai` lineage on the Mac; GitHub Release with the generated green
matrix (`release_matrix.py --version 0.33.3-dynres runs/full-0333.json`, full
run only — a later `--skip-pinned` smoke would overwrite green with
*skipped*); README **Current fold** pointer; deploy as
`ollama-0.33.3-dynres-<n>-g<sha>` built from the tag; `:11434` stays parked.

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

1. ☐ D1 and D2 decided and recorded (ADR where a recorded decision changes).
2. ☐ Housekeeping PRs merged: #211; `constrain.go` deletion; #201/#213/#210; `release_matrix.py` quality column; 0.33.2 tag hygiene.
3. ☐ Merge with the resolutions above; zero conflicts; Gate 1 grep clean.
4. ☐ Gate 2 green (`go build`, `go vet`, the listed `go test` set, `test_verdicts.py`).
5. ☐ Gate 3 image built on `bigdisk`, patch/payload/go-license proofs recorded.
6. ☐ Gate 4: CUDA ladders re-measured then `payload_pin` moved to `0f3a71be1`; full preflight exit 0 with `poison_probe` corroborated; `mlx-metal-0-33-3` profile cut and PASS; goldens PASS (or recalibrated with a bf16 control, as #225).
7. ☐ Gate 5: five think-off nvfp4 cells reproduce to three decimals; MMQ padding gate and qwen2.5vl probe re-run at b10760.
8. ☐ `v0.33.3-dynres` (+ `-maxusai`) tags on the merge commit; Release with generated matrix; README pointer; SPEC H11 note in the merge-commit body.
9. ☐ Deployed as `ollama-0.33.3-dynres-…` from the tag; `sync-0.33.2` image retained as rollback.

## Effort

Housekeeping ½ day · merge + ports 1–2 days · image ~3 h · CUDA preflight
~1 h (+quality) · Metal profile + goldens ½ day · Gate 5 spot-check ~4 h GPU.
D1-B adds a 2–3 day spike before its own ADR.
