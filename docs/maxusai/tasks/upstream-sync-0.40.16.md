# TASK: fold upstream v0.40.16

Upstream [v0.40.16](https://github.com/ollama/ollama/releases/tag/v0.40.16) (2026-11-09) folds onto `main` at
`63542f063`: the v0.40.13-dynres tag, 908's retirement (#505), and the docs and Metal preflight that followed (#506,
#507).
- **Size:** v0.40.13..v0.40.16 is 28 commits, 151 files, +15,582/−1,627. It takes v0.40.14 and v0.40.15 on the way.
- **Where:** branch `task/upstream-sync-0.40.16`, on the CUDA host (`ai-server/mlx-cuda`).
- **Why the tag and not v0.40.17:** v0.40.17 is at rc0 (2026-11-10). It is a structural fold of its own and should not
  share gates with this one:
  - Its llama.cpp, b12257, removes `tools/mtmd/clip.cpp` and `clip.h` (7d7f02875, code consolidated into the mtmd
    files). 002, 004 and 801 then fail for missing files, about 130 changed lines of the fork's vision core to re-cut
    by hand. Upstream's own 001 is already re-cut there.
  - Its MLX concurrent generation (ab9295d03) conflicts with the fork in 9 files, `mlxrunner/pipeline.go`,
    `runner.go`, `server.go` and the gemma4 models among them.
  - Folding v0.40.16 first keeps attribution clean: if gemma4 scores move in this fold, they moved with v0.40.16's
    gemma4 changes (below), not with the mtmd refactor.
- **What upstream brings:**
  - models: MiMo V2.6, the AutoTrust decision models (Apple silicon), Ripple Preview;
  - recurrent and vision GGUF models may run more than one request at a time (84dc81db1, the num_parallel=1 override
    removed from `server/sched.go`);
  - the OpenAI-compatible endpoints stop filling in sampler options the client left out (c1db5f9ef);
  - MLX: YaRN scaling in Qwen3 and Qwen3.5 (cdd4b55b4), chunked top-k for large vocabularies, the generation cache
    closed on shutdown, bundled CUDA libraries preloaded by full path on Linux (ddf22489b), unsupported CUDA devices
    refused at runner start (2e7b870fb), Qwen3.8 Flash Next memory on Apple silicon;
  - `ollama launch` and app work (Claude Code, Codex, Kimi Code, Pi, ChatGPT), which the server does not use.

## Status (2026-11-14)

| gate | state |
|---|---|
| 1, the merge | **done**: `7c25e6053`, **no conflicts**. `go build`, `go vet` and `go test ./...` pass in `golang:1.26` (go1.26.10, 77 packages ok). The MLX tests skip without libmlx, so they say nothing about MLX |
| 2, docs and paths | **clean**: `check_source_paths.py --changed-since 63542f063` resolves every reference |
| 3, the patch series | **done**: `d96be0a54`. All eight patches apply in order to a fresh b12177 with plain `git apply`. 802, 805 and 903 are re-cut; their changed lines are byte-identical to the v0.40.13 cut (below). 903 compiles for sm_75 and sm_120a, and its guard fires on a mutation in all 10 config tables, which are all of b12177's (`tasks/mmq-compile-0416.sh`). 903's guard-page gate (`tasks/mmq-0416-gate.sh`) **passes on sm_75 and sm_120a**: every pass is backed by a faulting control, and the ids_dst control faults in both new fused-NVFP4 cases (below) |
| no-GPU harness gates | **green**: `test_verdicts.py`, `test_summarizers.py`, `test_rescore.py` (1 skipped, as in CI), `test_mlx_test_gate.py` |
| 4, image | **gfx1151: built**, `maxusai-ollama:0.40.13-dynres-8-ga774f7a-rocm7-gfx1151` at `a774f7af5`, llama-server `6c01da627`; hipcc compiles 903's guard and its mutation fires in all six AMD tables ("gfx1151" below). CUDA: **building** (`build-0416.sh`, from 2026-11-14 18:11 at `a774f7af5`, stamp `0.40.13-dynres-8-ga774f7a`). Old scratch was deleted first, which took the 8TB array from 36 GB to 66 GB free |
| 5, preflight | **gfx1151: passes** on the new `rocm7-0-40-16` (PASS=27 SKIP=5); all 24 ladder values unchanged. CUDA: not run. The canary reads a run-local expectations copy (`tasks/exp-0416.py`, below) |
| 6, campaigns | **gfx1151:** think-off on four models against b11963, and 84dc81db1's slot change split from the payload on qwen3.8 and nemotron3 ("gfx1151" below). CUDA, Metal: not run |
| tag and deploy | not tagged, not deployed |

## Native inputs

| input | v0.40.13 fold | v0.40.16 |
|---|---|---|
| `LLAMA_CPP_VERSION` | b11963 | **b12177** (`6c01da627`; b12111 at v0.40.15) |
| `MLX_VERSION` | `065ec468` | **`54a70511`** (v0.40.15) |
| `MLX_C_VERSION` | `57731b00` | `57731b00`, unchanged; upstream adds the carry patch `0002-cuda-get-arch-list.patch` for 2e7b870fb |

- **MLX `54a70511`** is 12 commits. On CUDA, the one that can move output is 946077b5f, which keeps `g` and `beta` in
  float32 in `gated_delta_update`: the recurrent layers of qwen3.5/3.6 and nemotron on MLX. 54a70511 itself exposes
  the CUDA compilation targets that 2e7b870fb checks. The rest are Metal fixes (convolution bounds, grid truncation
  past 2^27 rows, quantized kernel cleanup), a strided-scan out-of-bounds fix, Windows DLL loading and docs.
- **llama.cpp b11963..b12177** is 214 commits. Those that touch what the fork serves:
  - **7bd180830, gemma4 resizes without letterbox padding.** `image_resize_pad = PAD_NONE` for the gemma4
    projectors. gemma4's preprocessor (`mtmd_image_preprocessor_dyn_size`) resizes to the size 004's
    `calc_size_budget_fill` returns. That size floors each axis to the patch grid, so its aspect ratio is not the
    source's, and until now the resize letterboxed the difference with black (`PAD_CEIL`). Now it stretches to the
    grid, as transformers' `image_processing_gemma4.py` does. The token counts do not change; the pixels in them do.
    This is the class of change "diff the projector cases at each pin move" is for: **gemma4 GGUF vision scores and
    goldens may move**, bounding boxes most of all.
  - 6cb3e672e, gemma4 graph optimizations (prompt processing on E2B and E4B); 344846b4f, gemma3n/gemma4 create
    `token_embd` before the tied output; e853ec9a8, the Gemma4 assistant follows flash attention.
  - 1b6935b22, `flash_attn_ext_rows` (unified KV for multi-sequence), the first CUDA flash-attention change since
    908's retirement; 78febbad9 enables sparse attention on CDNA.
  - 69ee2ecc0, CUDA MMQ/MMF with a linearised MoE data layout, and 5ce06fc71, fused NVFP4 gate/up+GLU in MMQ: both
    in the code 903 guards.
  - Nothing in `tools/mtmd` for nemotron's or qwen3.5's image path; the other mtmd commits are TTS, video, InternVL
    and a cmake unity build.

## Gate 3: the patch series at b12177

| patch | at b12177 |
|---|---|
| 001 hooks, 002 nemotron dynres, 004 gemma4 budget fill, 005 pinned overshoot, 801 clip meter | apply unchanged |
| 802 LM node meter | **re-cut**: its `#include` hunk's context moved in `src/llama-context.cpp` |
| 805 FA force-tile | **re-cut**: its `#include` hunk's context moved; `fattn.cu`'s guard is now MUSA-only (78febbad9) |
| 903 MMQ padding guard | **re-cut**: the padding hunk's context moved (upstream simplified the MoE `ncols_opt` estimate above it) |

Each re-cut patch's `+`/`-` lines diff empty against the v0.40.13 cut, so the re-cuts change context only. The full
series was then applied in order to a fresh `git archive b12177` tree.

**903 at b12177.** The two MMQ `src1` allocations (dense, and MUL_MAT_ID) both still add `src1_q8_1_padding`. The
fused NVFP4 gate/up path runs inside `ggml_cuda_mul_mat_q` and takes the same allocation. MUL_MAT_ID's
`src1_sorted` buffer in `ggml-cuda.cu` (f32 or f16) belongs to the fallback after MMQ and MMF decline; the MMQ
branch returns before it, so the MMQ y load never reads it.
- **Compile gate** (`tasks/mmq-compile-0416.sh`, GPU-free, CUDA 13.0): sm_75 and sm_120a compile clean. A mutation
  dropping the helper's padding to `nthreads` ints fails both builds on the static_assert, in all 10 tables
  (`ampere blackwell cdna gcn pascal_dp4a pascal_older rdna2 rdna3 rdna3_5 rdna4`), with no syntax errors.
- **Guard-page gate** (`tasks/mmq-0416-gate.sh`): the v0.40.13 gate on b12177 + 805 + 903. Its debug rewrite is
  re-anchored on b12177's `prec_src1` line, where the fused gate now carries its own `src1` precision. Every variant
  and control was dry-run on the tree first (`mmq-variant-0416.py`'s own checks). sm_120 gains two cases for
  5ce06fc71's fused NVFP4 path: the plain op, which takes the native FP4 y block, and its W4A8 twin, which forces q8_1
  y.

**903's guard-page gate on b12177** (`tasks/mmq-0416-gate.sh`, run by one driver on 2026-11-14: sm_75 on GPU1, RTX
2080 Ti, 18:03-18:42; sm_120a on GPU0 18:42-19:34; results `tasks/mmq-successor-results/gate-0416-sm75.tsv` and
`gate-0416-sm120.tsv`). The variants are v0.40.13's: `pr462` (the branch), `pr462jb` (src1 padded by J_best blocks)
and `pr462ids0` (no ids_dst padding).

| case | pr462, all modes | jb control, guard:src1 | ids0 control, guard:ids_dst |
|---|---|---|---|
| dense321 (sm_75) | pass | **faults 3/3** | n/a (no ids_dst) |
| ids16 (both) | pass | **faults 3/3** | **faults 3/3** |
| j100_b0 (sm_120a) | pass | **faults 3/3** | **faults 3/3** |
| j100_b1 (sm_120a) | pass | **faults 3/3** | **faults 3/3** |
| fu_dense17 (both) | pass | passes | n/a (no ids_dst) |
| fu_id17b (both) | pass | passes | **faults 3/3** |
| fu_id2048 (both) | pass | passes | **faults 3/3** |
| fu_nvfp4_id150 (sm_120a, new) | pass | faults 1/3 | **faults 3/3** |
| fu_nvfp4_w4a8_id150 (sm_120a, new) | pass | passes | **faults 3/3** |

- **`pr462` passes in every case on both GPUs:** 3/3 under the src1 guard, 3/3 under the ids_dst guard and 3/3 on
  the stock pool, plus 1/1 at exact allocation with 0 memcheck errors. That is 20 case-mode rows on sm_75 and 32 on
  sm_120a. No GPU process was left behind.
- **Evidence for src1:** dense321, ids16 and the j100 cases. **Evidence for ids_dst:** every MUL_MAT_ID case,
  including both new NVFP4 cases.
- The fused src1 rows are not evidence, because their control passes too, as at v0.40.13. Neither is the fused NVFP4
  src1 row, whose control faults only 1/3.
- The `MMQ_FUSED` debug print confirms that every `fu_` case (three on sm_75, five on sm_120a) reaches the fused MMQ kernel, and that ids16 and the
  j100 cases do not.
- **Changed since v0.40.13:** the j100 src1 controls fault 3/3 here, against 1/3 and 2/3 at b11963.
- The gate's summary tables label the first column with v0.40.13's tree name (b11351). That label is a carried
  string: each run's start line records the tree it actually ran (`6c01da627`).

## What the gates must re-measure

- **gemma4 GGUF vision** (7bd180830, 6cb3e672e): ladders should hold, since the token counts do not change. Scores and
  goldens may move: preflight, then a gemma4 campaign against production's runs.
- **gemma4 GGUF loops** (1b6935b22 in flash attention, with 908 gone): think-on on 26b and 31b, the full ladder.
- **MoE and NVFP4 GGUF on CUDA** (69ee2ecc0, 5ce06fc71): qwen3.6 MoE and the nvfp4 GGUFs, in preflight and on speed.
- **MLX recurrent models** (946077b5f): the `mlx-cuda` rows for qwen3.5/3.6 and nemotron may move.
- **The MLX runner start** (2e7b870fb, ddf22489b): the canary's first MLX load proves the preload and the device
  check on the deduplicated payload (`mlx_cuda_v13` + `cuda_v13`).
- **`/v1` sampling** (c1db5f9ef): clients that send no temperature or top_p now get the model's defaults instead of
  1.0/1.0. The harness uses `/api/generate` and does not see it; `/v1` consumers will.
- **num_parallel** (84dc81db1): CUDA production leaves `OLLAMA_NUM_PARALLEL` at 1, so the removed override changes
  nothing there. **gfx1151 production runs `OLLAMA_NUM_PARALLEL=2`**, so there qwen3.8 and nemotron3 go from one slot to
  two; measured below.

## gfx1151 (ROCm host, 2026-11-14)

The image is `maxusai-ollama:0.40.13-dynres-8-ga774f7a-rocm7-gfx1151`: #508 at `a774f7af5`, built with
`Dockerfile.rocm` on `rocm/dev-ubuntu-24.04:7.2.4-complete`, llama-server `6c01da627`. Later commits on #508 are docs
and task records. Every run used a canary in production's environment (f16 KV, flash attention on, two-pass,
`OLLAMA_NUM_PARALLEL=2`) on a hard-linked clone of production's store, with each model as production serves it
(gemma4:26b converted in the clone, as in every baseline).

- **903 under hipcc** (the toolchain stage of `Dockerfile.rocm`, `GPU_TARGETS=gfx1151`): a clean compile; the padding
  mutation fails on the guard's `static_assert` in all six AMD tables (`gcn`, `cdna`, `rdna2`, `rdna3`, `rdna3_5`,
  `rdna4`), with no syntax errors; restored, it compiles. b12177's new `RDNA4_M` family (eb4e244be) reuses the
  `rdna3_5` table in both `ggml_cuda_mmq_get_config` overloads, so it is covered.
- **Gate 5, `rocm7-0-40-16`.** `measure_ladder.py` and `aspect_measure.py`: all 24 values and the three prefixes equal
  `rocm7-0-40-13`'s, unchanged since b11351. 7bd180830 moves no gemma4 token count (004 already resizes directly).
  Preflight `--quality`: VERDICT PASS, PASS=27 SKIP=5 (`preflight/runs/preflight-rocm7-0416-canary-ga774f7a.json`),
  run-local copy as for the earlier folds.
- **Gate 6, think-off against b11963** (`r0413a_`), the same suite tree and settings, tag `r0416a_`. Every arm finishes
  all 27 cases.

| model | cells moving | quality moves (b12177 : b11963 : neutral) | what moved |
|---|---|---|---|
| nemotron3:33b-q4_K_M | 90 of 1010 | 24 : 17 : 21 | spread, both ways |
| gemma4:26b-a4b-it-q4_K_M | 40 of 1007 | 0 : 5 : 1 | `bbox_contract_perobject` flips to y-first (drops the contract) |
| gemma4:31b-it-q4_K_M | 39 of 1013 | 1 : 4 : 1 | `bbox_contract_pinned` flips back to y-first, as it had before b11963 |
| qwen3.8:27b-q4_K_M | 39 of 1005 | 0 : 7 : 5 | small IoU drops (0.01–0.03; `bbox_contract_real_1img` 0.995 → 0.965) |

  The gemma4 rows are the dialect flips these two cases have made at every payload move, both ways (the v0.40.5 and
  v0.40.13 records). No case loses JSON validity.

- **84dc81db1 on gfx1151, the slot change split from the payload.** qwen3.8 and nemotron3 re-ran at
  `OLLAMA_NUM_PARALLEL=1` on the same build (`r0416p1_`):

| model | slots 1 → 2, same build | b11963 → b12177, both at 1 slot |
|---|---|---|
| nemotron3:33b-q4_K_M | 3 of 1009 cells (one IoU ±0.002, one answer 11 tokens shorter); 0 quality moves | 89 cells, 24 : 17 |
| qwen3.8:27b-q4_K_M | 20 of 1005 cells; 2 : 2 | 41 cells, 1 : 9 (the IoU drops above, plus one `scene_single_pinned` draw, 0.964 → 0.731, fine at 2 slots) |

  So the moves are the payload's; the second slot is close to neutral.

| at `num_ctx` 8192 | loaded size, 1 slot → 2 | KV (f16) | recurrent state | llama-server |
|---|---|---|---|---|
| nemotron3:33b-q4_K_M | 25.40 → 25.50 GB | 48 → 96 MiB | 48 → 95 MiB | `-c 8192 -np 1` → `-c 16384 -np 2` |
| qwen3.8:27b-q4_K_M | 17.09 → 17.78 GB | 512 → 1024 MiB | 150 → 299 MiB | same |

  The KV half scales with `num_ctx`: at qwen3.8's 65536 rung the second slot costs about 4 GiB of KV more. A
  concurrent pair of requests (greedy, 160 tokens) finishes `stop` on both with identical text on each model: 2.8 s
  for nemotron3, 13.7 s for qwen3.8.
- **Not run here:** think-on.

## Canary expectations

The fold's interim builds stamp `0.40.13-dynres-N` (`main`'s last tag is `v0.40.13-dynres`), which `cuda-0-40-13` and
`mlx-cuda-0-40-13` claim, and their `payload_pin` would fail a b12177 build. `tasks/exp-0416.py` writes a run-local
copy that moves the 0-40-13 CUDA pair to `cuda-0-40-16` and `mlx-cuda-0-40-16`, with the b12177 / `54a70511` pins,
908 out of the patchset, and a version pattern admitting the interim stamp. The rows stay 0-40-13's until
`measure_ladder.py` says otherwise. The checked-in profiles come after measurement, as at v0.40.13.

## Open

- **Disk.** The `bigdisk` buildkit cache was 82.9 GB before this build: 44.6 GB from the v0.40.13 fold's builds
  (2026-11-03), 32.8 GB from 2026-10-13, and 5.4 GB older. A full build at a new llama.cpp and MLX pin adds roughly
  what the last fold's did, so the next fold needs space again.
- Gates 4-6 on CUDA. The canary (`tasks/canary-0416.sh`) is chained to the image. Metal's are its own; gfx1151's are
  above.
