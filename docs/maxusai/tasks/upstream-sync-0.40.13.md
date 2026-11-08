# TASK: fold upstream v0.40.13

Upstream [v0.40.13](https://github.com/ollama/ollama/releases/tag/v0.40.13) (2026-10-28) folds on top of the v0.40.5
fold, #498, at `529db8646`. #498 was unmerged when this fold started, and its head includes #500 (the ROCm clef COPYs).
- **Size:** v0.40.5..v0.40.13 is 52 commits, 292 files, +31,241/−1,457. It takes v0.40.6 through v0.40.12 on the way.
- **Where:** branch `task/upstream-sync-0.40.13`, on the CUDA host (`ai-server/mlx-cuda`). It stacks on
  `task/upstream-sync-0.40.5`. #498 should land as a merge commit, not a squash; a squash drops the v0.40.5 ancestry,
  and this branch would re-conflict on all of v0.40.3..v0.40.5.
- **Why the tag and not `upstream/main`:** the 7 commits after v0.40.13 (launch, openai, app, the MLX chunked top-k,
  a buildx pin) merge with the same 11 conflicts, byte-identical apart from the marker labels. Nothing in them is
  needed.
- **What upstream brings:**
  - models: pplx-decider, Flint, LFM3 (text, VL, audio, omni), the d1/d3 decision models, the DFlash2 draft model and
    Phonon TTS;
  - Gemma 4 prefill on Apple silicon (MLX bump), and more detail kept in small images on the compact towers;
  - llama-server infers the draft speculative type;
  - the scheduler reads loaded models without `refMu` (`runner.model` becomes atomic);
  - the CUDA runtime payloads are deduplicated (2cd63f7a1);
  - Go 1.26.10.

## Status (2026-11-04)

| gate | state |
|---|---|
| 1, the merge | **done**: `88529ea2e`, 11 conflicted files, 18 hunks (below). `go build`, `go vet` and `go test ./...` pass in `golang:1.26` (go1.26.10). The MLX tests skip without libmlx, so they say nothing about MLX |
| 2, docs and paths | **clean**: `check_source_paths.py --changed-since 529db8646` resolves every reference |
| 3, the patch series | **done**: `c32183d9d`. All ten patches, upstream's `002-metal` included, apply in order to a clean b11963. 004 and 903 are re-cut. 903 compiles for sm_75 and sm_120a, and its guard fires on a mutation in all 10 tables; on gfx1151 (hipcc, ROCm 7.2.4) the same mutation fails in all 10 tables, `rdna3_5` included (#501). 903's guard-page gate (`tasks/mmq-0413-gate.sh`) **passes on sm_75 and sm_120a**: every pass is backed by a faulting control. b11963's fused gate/up MMQ path is covered by three new cases, and its ids_dst control faults there (below) |
| no-GPU harness gates | **green**: `test_verdicts.py` (269), `test_summarizers.py`, `test_rescore.py` (1 skipped, as in CI), `test_mlx_test_gate.py` |
| 4, image | **built on CUDA**: `maxusai/ollama:sync-0.40.13-fold`, stamp `0.40.3-dynres-20-g81dda7d`, 4 h 16 min (a full native build: all three pins move; host load 30-116 from other jobs); llama-server `39febd2ad`. It predates the review fixes below (Go only: the qwen3_5 cap, a `logMu` write). **gfx1151: built**, `maxusai-ollama:0.40.3-dynres-21-g833a5d2-rocm7-gfx1151`, llama-server `39febd2ad` |
| 5, preflight | **CUDA: both pass** on the new `cuda-0-40-13` (PASS=20 SKIP=9) and `mlx-cuda-0-40-13` (PASS=11 SKIP=18), in a production-like canary (`--gpus all`): all six ladders byte-identical to the 0-40-5 rows, payload pins `39febd2ad` and `065ec46`, 0 cudaMemcpy errors, 0 OOMs (`runs/preflight-*-0413-canary-g81dda7d.json`). **gfx1151: passes** on the new `rocm7-0-40-13` (PASS=27 SKIP=5); all 24 ladder values equal the b11669 and b11351 rows ("gfx1151" below) |
| 6, campaigns | **CUDA: 908's loop-rate gate** on b11963: 3 vs 3 loops with and without 908, and think-off no longer favours 908 ("908's loop-rate gate on b11963" below); nemotron3's format probe (below). The other think-off cells are not run. **gfx1151:** think-off on four models against b11669; nemotron3's move is the payload's, split by two hybrid images ("gfx1151" below) |
| tag and deploy | **tagged** `v0.40.13-dynres` on `85d636818` (#501's merge). **Deployed:** gfx1151 2026-11-04 08:57 (#504), CUDA 2026-11-04 10:09 ("Deploy on CUDA" below). 908 ships in this tag and is retired on `main` after it (#505). Metal: not run, not deployed |

## Native inputs

| input | v0.40.5 fold | v0.40.13 |
|---|---|---|
| `LLAMA_CPP_VERSION` | b11669 | **b11963** (b11738 at v0.40.7, b11803 at v0.40.9, b11845 at v0.40.11, b11901 at v0.40.12) |
| `MLX_VERSION` | `bebd6422` | **`065ec468`** (v0.40.12) |
| `MLX_C_VERSION` | `a341b492` | **`57731b00`** (v0.40.12) |

- **MLX-C `57731b00`** ("bump to v0.32.4") has qmm's `global_scale` and `mlx_fast_gated_delta_update` natively. So
  upstream deletes its two MLX-C carry patches and carries one new one, `0001-mlx-c-sdpa-window-size.patch`. The
  fork carried neither of the old two, so nothing of ours goes.
- **MLX `065ec468`:**
  - The CUDA vector SDPA kernel now runs at head_dim 256 (a82b8fb5), which covers gemma4's sliding-layer decode.
  - Sliding-window SDPA (360078a8) is composed, not fused, on CUDA. Upstream's 66ab8482e dispatches sliding-window
    prefill through it on every platform.
  - Large-MoE indexing fixes: gather_qmm past 2^32 elements (53c18705), and grouped gemm, gather/scatter and pad
    above 2^31 elements.
  - No nvfp4/mxfp4, gated-delta or allocator changes on CUDA, and nothing for device selection.
  - The CUDA completion-worker busy loop fix (ce916dbb, the idle runner pinning a core) was already in `bebd6422`.
- **The CUDA payload layout changes (2cd63f7a1).** The `mlx_cuda_v13` overlay keeps only MLX's own libraries; the
  shared CUDA libraries go to the sibling `cuda_v13`, with RPATH `$ORIGIN/../cuda_v13`. `Dockerfile` needs no edit
  (the final COPY overlays the MLX stage's `cuda_v13` onto llama-server's). A standalone MLX tool now needs both
  directories on `LD_LIBRARY_PATH`, and a library-swap gate must account for the shared `cuda_v13`.

## The conflicts and their resolution

| file | upstream cause | resolution |
|---|---|---|
| `cmake/mlx/CMakeLists.txt` (2) | 2cd63f7a1 | upstream's. It sets `$ORIGIN` plus `$ORIGIN/../cuda_vXX` and keeps quadmath, so the fork's RPATH hunk (`f2ad5c6a4`) retires |
| `mlxrunner/client.go` | 2cd63f7a1 `libraryPathEnv` | both, next to the fork's `budgetWithOverride` |
| `mlxrunner/model/gemma4/gemma4.go` (2) | f52b9e742 (DFlash on gemma4) | the fork's vision fields plus upstream's `auxHiddenLayers` capture; the fork's `layer.Forward` call |
| `mlxrunner/model/gemma4/process_image.go` | d44c25819 (modify/delete) | stays deleted: D1-A sizes through `llm.BudgetFillSize` (ADR 0021). Upstream's round-up for the compact towers (E2B, E4B) is not taken; it applies to neither 12b, 26b nor 31b |
| gemma4 `media_test.go` (3), `vision_test.go` (1) | d44c25819 | the fork's |
| gemma4 `vision.go` | 66ab8482e (SDPA gains a window) | the fork's attention, passing `mlx.FastSDPANoWindow` |
| `mlxrunner/model/qwen3_5/process_image.go` (3) | 8be580ad0 (config `VisionMaxPixels`) | the fork's `smartResizeBounded`; `smartResize(..., maxPixels)` wraps it. `pixelBudget` treats `VisionMaxPixels` as a ceiling: a request budget can lower it, never raise it, and `min_tokens` cannot push the floor over it (`TestPixelBudgetKeepsTheCheckpointCap`). Without that, pplx-decider and d3 would get the 16.7 MP default with no conflict marker; d1 is not qwen3_5 and caps itself (`lfm3.CapImagePixels`) |
| `mlxrunner/speculate.go` | 2506f851e (DFlash2 fixed depth) | both: `limit = capped(scheduled)`, and for a BlockSelector `limit = capped(dflash2Depth)` with `fixedDepth`. Whether DFlash2 tolerates a `draft_num_predict` below 7 is not verified |
| `server/sched.go` (3) | 52f168120 (`runnerRef.model` is an `atomic.Pointer`) | `unload` stores nil under `logMu`. `needsReload` reads `loaded := runner.model.Load()` and keeps the fork's two MLX num_ctx clauses on it. `LogValue` keeps the fork's `logMu` read. The fork-only post-load MLX refresh uses `Load()` |
| `server/sched_test.go` | 52f168120 | upstream's, with the fork's `runnerRef{model: …}` literals converted (and one in `routes_mlx_numctx_test.go`) |

Merged without a marker, and checked:
- `server/images.go`: upstream's new renderer `gemma4-nemotron` (f52b9e742) is added to `isGemma4Renderer`
  (`81dda7d30`), so a gemma4 safetensors model on it still suppresses audio.
- `server/routes.go`: `parserNameForModel` and the `OLLAMA_FORMAT_TWO_PASS` path are untouched.
- `llm/`: e51196fe8 omits `--spec-type` for an unknown external draft. 3fbc609be maps `repeat_last_n=-1` and
  extends the 2048-ctx floor to any model with projectors. Neither touches `kv_cache_type`, `--cache-ram` or the
  `-b/-ub` launch (ADR 0036).
- ROCm: upstream changed nothing in its ROCm stages, CMake presets or `envconfig`, so `Dockerfile.rocm` (ADR 0042)
  needs nothing beyond #500.

## Review of the merge

Two reviews, one on this host and one on the ROCm host (#501), found no bug introduced by the merge:
- `sched.go`: every `runner.model` access is a `Load()`/`Store()`, nil is handled, and upstream's other changes in 52f168120 are kept.
- `speculate.go`: a cap below 7 on a DFlash2 draft is safe. Upstream already drafts shorter blocks near the end of `num_predict`, and every shape follows the block length.
- gemma4: the DFlash taps are reachable after the fork's `Forward`.

Fixed after review (`server/sched.go`, `mlxrunner/model/qwen3_5/vision.go`):
- The qwen3_5 cap above, which a request budget could bypass. It is latent today, because the decision models score through `PrepareMedia` with no budget.
- A pre-existing fork race: the post-load MLX served-context write to `Options.NumCtx` held `refMu` only, while `LogValue` reads `Options` under `logMu`. It now takes `logMu`.

Not covered by a test: a capped request on a selector draft (`maxLimit=3` should give `limit 3` with `fixedDepth`). The natural home is next to upstream's `TestSelectorDraftKeepsFixedDepth`, which is mlxtest and skips without libmlx.

## Gate 3: the patch series at b11963

Applied with plain `git apply`, in `compat.cmake`'s order, to a clean b11963 (`39febd2ad`):

| patch | at b11963 | why |
|---|---|---|
| 001 hooks | applies, offsets | |
| 002 nemotron dynres | applies, offsets | still needed: `mtmd.cpp` keeps `fixed_size` for `NEMOTRON_V2_VL`, which the fork's nemotron3 checkpoint still maps to |
| 002-metal-unary-cast-constexpr (upstream's, 04088e7c9) | applies | Metal only. It shares the 002 prefix and sorts after ours; the two touch different files |
| **004 gemma4 budget fill** | **re-cut** | f711c8f30 moved dyn_size sizing into the static `get_target_size()`; the budget-fill branch moves there |
| 005 pinned overshoot | context only | |
| 801, 802, 805 | apply, offsets | |
| **903 MMQ y-tile guard** | **re-cut** | c49edf096 moved tile choice into `ggml_cuda_mmq_get_J_best()`. Its out-parameter now reports the chosen config's padded y tile, so `src1_q8_1_padding` stays `ggml_cuda_mmq_get_nbytes_y_tile()`, equal to upstream's `GGML_PAD(J*144, nthreads*4)`. The allocation does not change. Upstream still has no guard |
| 908 FA tiling revert | applies, offsets | still needed: nothing in b11669..b11963 touches `fattn-mma-f16.cuh`; ce8caa6e6's tiling is intact |

The tree from the re-cuts and the tree from applying the committed patches are byte-identical (`diff -rq`).
`tasks/mmq-compile-0413.sh`, GPU-free, nvcc 13.0, `--generate-code` per architecture:

| arch | mmq.cu | mutation: helper without the nthreads padding |
|---|---|---|
| sm_75 | compiles | fails on the guard's `static_assert`, all 10 tables named, 0 syntax errors |
| sm_120a | compiles | the same |

**903's guard-page gate on b11963, sm_75** (GPU1, RTX 2080 Ti; `tasks/mmq-0413-gate.sh`, results
`tasks/mmq-successor-results/gate-0413-sm75.tsv`):
- Variants: `pr462` (the branch), `pr462jb` (src1 padded by J_best blocks) and `pr462ids0` (no ids_dst padding). Each mutation applies exactly once, and the build asserts it. The debug-allocator rewrite is re-anchored on c49edf096's `prec_src1 = gate ? ...` line.
- New cases: three of b11963's own `MUL_MAT_VEC_FUSION` cases (`fu_dense17`, `fu_id17b`, `fu_id2048`). An `MMQ_FUSED` debug print confirms each reaches the fused MMQ kernel.

| case | pr462, all modes (3 runs each; exact 1, memcheck 0 errors) | jb control, guard:src1 | ids0 control, guard:ids_dst |
|---|---|---|---|
| dense321 | pass | **faults 3/3** | n/a (no ids_dst) |
| ids16 | pass | **faults 3/3** | **faults 3/3** |
| fu_dense17 | pass | passes (J_best padding already covers it) | n/a |
| fu_id17b | pass | passes (this routing does not reach the worst case) | **faults 3/3** |
| fu_id2048 | pass | passes (128 blocks vs 127 needed) | **faults 3/3** |

Evidence: src1 on dense321 and ids16; ids_dst on ids16, fu_id17b and fu_id2048. That matches the v0.40.5 result. The fused src1 rows are not evidence, because their control passes too.

**sm_120a** (GPU0, after the canary, 2026-11-03 15:47-16:44; `gate-0413-sm120.tsv`):

| case | pr462, all modes | jb control, guard:src1 | ids0 control, guard:ids_dst |
|---|---|---|---|
| ids16 | pass | **faults 3/3** | **faults 3/3** |
| j100_b0 | pass | faults 1/3 | **faults 3/3** |
| j100_b1 | pass | faults 2/3 | **faults 3/3** |
| fu_dense17 | pass | passes | n/a (no ids_dst) |
| fu_id17b | pass | passes | **faults 3/3** |
| fu_id2048 | pass | passes | **faults 3/3** |

`pr462` passes 3/3 under both guards and on the stock pool in every case, and 1/1 at exact allocation with 0 memcheck
errors. The j100 controls fault at a rate, as at v0.40.5, so ids16 carries the src1 evidence there. No GPU process was
left behind.

Still open upstream, so carried: the qwen2/2.5-VL fp16 overflow (ollama#18070), and the image chunk vs ubatch
assert (llama.cpp #28954 capped `image_max_tokens` to `n_ubatch` before b11669, but the assert remains, so ADR 0036's
batch sizing stays).

## What the gates must re-measure

- **gemma4 GGUF vision.** llama.cpp 5ba3b9fd3 (#30882, in b11963, not in b11669) forces `FFN_GELU` on GEMMA4V and
  GEMMA4UV. The library gemma4 GGUFs carry no `clip.use_gelu`, so on b11669 their tower ran gelu_quick. Every
  gemma4 GGUF vision cell moves: rerun the vision goldens and bbox cells.
- **Qwen-VL geometry.** f711c8f30 puts QWEN2VL/25VL/3VL on `PAD_NONE` with round-half-even sizing. Rerun the qwen2.5vl
  and qwen3-VL bbox cells.
- **Fused MMQ gate/up+GLU** (c49edf096, f5222f25c) for gemma4 and qwen3.6 quants on CUDA. `GGML_CUDA_DISABLE_FUSION`
  is the A/B knob.
- **CUDA and HIP kernel fixes:** the TILE FA loop bound (5aeac37a3; gfx1151 runs gemma4 on TILE), the FA vec
  shared-memory race (99db719bd), and HIP bf16 WMMA on gfx11/12 (9145ddb95). Also chunked CUDA graphs and faster
  low-bit MMVQ.
- **Grammar** (48c9f08b7): candidate rejection rewritten, with a parse recursion limit of 256. Rerun the format and
  two-pass tests.
- **Speculative and server:** SWA checkpoint invalidation (f174553e1); gemma4 is SWA, so recheck prefix-cache reuse.
  Also logprobs under speculation, and `n_pad` dropped from the KV memory classes (check the memory estimates).
- **MLX on CUDA:** gemma4 decode (head_dim 256 vector SDPA) and sliding-window prefill. Pin `num_ctx` and use a
  tail estimator.
- **908's loop-rate gate** again on b11963: gemma4:26b-a4b think-on on the full ladder, with and without 908.

## 908's loop-rate gate on b11963 (CUDA, 2026-11-03/04)

`tasks/loop908-0413.sh`, on the maintainer's word ("go ahead with the 908 gate"):
- **The arms:** the fold image (with 908) against the same image with `libggml-cuda.so` rebuilt from `81dda7d30` minus 908. That library is the only file in `cuda_v13` that differs; the new shared-library layout changed nothing there.
- **Setup:** a cold container per model on GPU0, production's env, `-b 2048 -ub 2048` throughout.

**Think-on, gemma4:26b-a4b, the full ladder to 131072.** Every capped test is a repetition loop (`loop_detect`: the deduplicated trace is 2-4% of the raw one):

| arm | not converged at 131072 |
|---|---|
| with 908 (`l0413w_`) | `scene_single_anchored`, `multi_3img_anchored`, `bbox_contract_adv_norm1` |
| without 908 (`l0413n_`) | `multi_3img_anchored`, `bbox_contract_adv_real`, `bbox_contract_adv_norm1` |

**3 against 3.** The history of the gate: b11081 1 vs 6 (sm_120), b11669 1 vs 2, b11963 3 vs 3. The script's summary
line counts NOT CONVERGED lines (one per ladder), not tests, and printed "1 / 1"; the table above counts tests.

**Think-off, `cmp_scored.py`,** quality moves read as favouring with 908 : without 908. Each cell is a single greedy run, so treat these as moves along one numerical path, not as rates:

| model | with vs without 908, b11963 | with 908, b11669 vs b11963 |
|---|---|---|
| gemma4:31b | 5 : 5 | b11669 6 : 3 |
| gemma4:26b-a4b | 0 : 6 (all in `bbox_contract_perobject`: with 908 it switches to y-first and drops the contract, IoU 0.05 vs 0.97) | b11669 8 : 3 |
| gemma4:e4b | 11 : 29, across 16 tests | b11669 21 : 10 |

**Reading.**
- 908's retirement condition is met on b11963: the pin's own tiling loops no more often than 908's.
- On think-off, 908 is neutral on 31b and behind on 26b and e4b.
- At b11669, 908 kept the 31b and 26b think-off cells byte-identical to production. On b11963 they move even with 908, so the payload moves gemma4 regardless (fused MMQ gate/up, the GELU switch).
- The think-on loop count rose with the payload: 1 to 3 with 908.
- Greedy GGUF on CUDA repeats byte for byte (the H100 campaign), so a plain repeat would reproduce these numbers. A sampled think-on rate (`THINK_TEMPERATURE`, n >= 3 on the five looping tests) is the check that would add evidence.

## nemotron3's structured-output move on CUDA (2026-11-03)

gfx1151's gate 6 found nemotron3 think-off moving 8:32 against b11669 (#503). Its hybrid images put the move on the
payload, and its format bisect found the three free-frame `bboxm_free_*` arms shape-unstable. One cell closed early
under `json` on b11963: `bboxm_free_anc_pos`, 83 tokens.

The same probe on CUDA (`tasks/fmtprobe-0413.py`, `tasks/fmtprobe-0413.sh`):
- the v0.40.5 fold's Go on both payloads: `sync-0.40.5-fold` (b11669) and the v0.40.13 image with that Go binary (b11963);
- a fresh container each, GPU0 only, production's env, `-b 1024`;
- the suite's own requests, greedy, think off, `num_ctx` 8192, `json` then no format.

| arm | `json`: b11669 | `json`: b11963 | no format: b11669 | no format: b11963 |
|---|---|---|---|---|
| `bboxm_pin_anc_named` (control) | 462, flat | 462, flat | 463, flat | 463, flat |
| `bboxm_free_anc_named` | 786, flat | 651, **nested** | 652, **nested** | 652, nested (byte-identical) |
| `bboxm_free_anc_pos` | 553, flat | **553, flat** | 554, flat | 554, flat |
| `bboxm_free_noanc_named` | 514, flat | 514, flat | 515, flat | 515, flat |

- **No early close on CUDA.** `bboxm_free_anc_pos` under `json` on b11963 answers in full (553 tokens, as on b11669). So
  gfx1151's 83-token close is not a backend-independent effect of the grammar rewrite (48c9f08b7).
- **b11669 nests too.** On CUDA, b11669 with no format produces the nested `bbox_2d` shape that gfx1151's b11963 suite
  cell showed. The nested-vs-flat choice on `bboxm_free_anc_named` flips with backend, payload and format. It is the
  arm's instability, as gfx1151 concluded, not a regression of b11963.
- Token counts agree across payloads in 7 of 8 cells. The texts differ in all but one, so the payloads are close
  numerically and not identical.

## Deploy on CUDA (2026-11-04)

Production `:11497` serves `0.40.13-dynres-0-g85d6368` since 10:09, at the maintainer's word.

**The image is a Go-only build, not a full build of the tag.** A full build started first:
- Its first attempt failed after 28 minutes on a transient `proxy.golang.org` stream error in `go mod download`.
- The second was stopped after 40 minutes, when the maintainer chose the swap. The native stages were not reusing the fold image's cache, so it faced about 3.5 more hours.

`tasks/swap-v0413.sh` puts the tag's binary on `sync-0.40.13-fold`:
- No native input changed between `81dda7d30` and the tag, checked before the build.
- The image `maxusai/ollama:v0.40.13-dynres` stamps the release version, and its binary hash matches the build.
- All 5,316 payload files are hash-identical to the gated fold image. llama-server reports `39febd2ad`.
- So a full build would differ in `/usr/bin/ollama`, as the v0.40.3 deploy's swap did. Its binary is sha256 `8b6938826a9d...`.

**The swap** (`tasks/chain-v0413-swap.sh`, with `deploy-v0402-point.sh`, the point-swap script of the last CUDA deploys, kept on the host) mirrored the live container, waited for five idle minutes and kept the old one stopped. Health checks passed: the right version, 63 tags before and after, `OLLAMA_KV_CACHE_TYPE=f16`, `OLLAMA_FORMAT_TWO_PASS=1`.

Rollback: `docker rm -f ollama-0.40.13-dynres-0-g85d6368 && docker start ollama-0.40.3-dynres-12-g8761533`.

**Production's own preflight did not run.** It needs about 56 GB free on GPU0 for its largest model and the reserve, and another workload left 30-36 GB for the whole 6.8 hours it waited. It was stopped at 17:00, so the README's `cuda` and `mlx-cuda` rows read "not run". The evidence on this payload is the fold image's canary (gate 5), whose Go is older: it lacks the review fixes and the later merges.

**The smoke** (`tasks/smoke-v0413.py`, 17:00) was the first inference on this binary, on the llama.cpp runner, `gemma4:e4b-it-q4_K_M`, then unloaded:

| request | wall | result |
|---|---|---|
| text | 302 s (cold load and kernel compile) | `stop`, "ready" |
| image | 57 s | `stop`, prompt 1,123 tokens, 54 generated, naming all six of the fixture's labels with their colours, unscored |

The runner launched with `-b 2048 -ub 2048` and f16 KV. The log shows no out-of-memory or `cudaMemcpy` error. It shows that one model serves text and images. It did not load an MLX model, so the tag's Go is unexercised on that engine on this host.

## Fork rows checked against v0.40.13 (retirement register)

All still carried; none of the retirement conditions is met:
- gemma4 vision on MLX through the fork's path: upstream still has no per-request image budget.
- MLX admission and memory/cache limits: upstream calls `SetWiredLimit` only.
- Stop sequences on MLX.
- `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR`: upstream's only drafting change is DFlash2's fixed depth.
- Transparent images over white.
- ADR 0014's `growOpeningChunk`.
- #487's single-GPU pin: neither MLX nor upstream Go selects a device.

The fork's RPATH/quadmath hunk in `cmake/mlx/CMakeLists.txt` is the one that converged.

## gfx1151 (ROCm host, 2026-11-03)

The image is `maxusai-ollama:0.40.3-dynres-21-g833a5d2-rocm7-gfx1151`: #501 at `833a5d25f`, built with
`Dockerfile.rocm` on `rocm/dev-ubuntu-24.04:7.2.4-complete`, llama-server `39febd2ad`. Later commits change Go on the
MLX and scheduler paths and docs only. The method is the v0.40.5 record's "gfx1151" section: a canary in production's
environment, on a hard-linked clone of production's store, with each model as production serves it.

- **Gate 5, `rocm7-0-40-13`.** `measure_ladder.py` and `aspect_measure.py` give all 24 values and the three prefixes
  identical to `rocm7-0-40-5`'s, so to b11351's too.
  - f711c8f30 moves no qwen35 token count at these geometries.
  - 004, re-cut into `get_target_size()`, still fills gemma4 to the budget.
  - Preflight `--quality`: VERDICT PASS, PASS=27 SKIP=5 (`preflight/runs/preflight-rocm7-0413-canary-g833a5d2.json`),
    with `payload_pin` `39febd2ad`, the pinned budgets, `think_format`, the quality floors at 1.00 and the poison canary.
- **903 under hipcc.** See gate 3 above: the mutation fails on the guard's `static_assert` in all six AMD tables.
- **Gate 6, think-off against b11669** (`r0405a_`, #502), with the same suite tree and settings. The fold's cells are
  `r0413a_`, all at `-b 2048`. Prompt token counts are identical in every cell, and every arm finishes all 27 cases
  with `stop`.

| model | cells moving, b11669 -> b11963 | quality moves (favour b11963 : b11669 : neutral) |
|---|---|---|
| nemotron3:33b-q4_K_M | 84 of 1010 | **8 : 32 : 17** |
| gemma4:26b-a4b-it-q4_K_M | 44 of 1007 | 2 : 4 : 1 |
| gemma4:31b-it-q4_K_M | 38 of 1013 | 7 : 0 : 1 |
| qwen3.8:27b-q4_K_M | 78 of 1005 | 29 : 9 : 10 |

- **qwen3.8 gains** (29:9), as f711c8f30's Qwen-VL geometry would predict. gemma4:31b gains too (7:0), and its
  `bbox_contract_pinned` now follows the contract. gemma4:26b moves both ways; that row flips back to `yxyx` there.
- **nemotron3 loses most of its moves (8:32).** 45 of its 84 cells are three free-format arms (`bboxm_free_anc_named`,
  `bboxm_free_anc_pos` and `bboxm_free_noanc_named`). On b11963 these answer with each box nested under a `bbox_2d`
  object. The coordinates are close to b11669's, but the scorer finds 0 of 6 boxes where it found 6. The rest are
  small IoU moves on anchored and scene cases, some up and some down.
- **The nemotron move is the payload's, not the fold's Go.** Two hybrid images were built from the two fold images,
  and each ran the same nemotron cell:
  - this fold's Go on the b11669 payload (`r0413hA_`): **0 of 1009 cells** differ from b11669's;
  - the v0.40.5 fold's Go on the b11963 payload (`r0413hB_`): **0 of 1010 cells** differ from this fold's.
- **Not yet bisected inside b11669..b11963.** The candidates are the grammar rewrite (48c9f08b7), which the
  structured-output arms pass through, and the mtmd changes nemotron_v2_vl shares (`build_inp_attn_mask`, the
  f16-mask assert in `build_attn`). A cell is not a verdict on the model: nemotron3's boxes are right in the nested
  answers, and the loss is how the answer is laid out. Think-on was not run.

## Open

- **Metal:** gates and deploy for this fold and the v0.40.5 one.
- **CUDA, production's own preflight on the tag:** `cuda-0-40-13` and `mlx-cuda-0-40-13` when GPU0 has about 56 GB free. It fills the README's two rows.
- **CUDA, before the next deploy of `main`:** the same two profiles on a no-908 build (#505). `sync-0.40.13-fold-no908` is that payload, and `IMG=maxusai/ollama:sync-0.40.13-fold-no908 RUN=canary-no908-g81dda7d tasks/canary-0413.sh` runs it. Queued once and stopped unrun.
