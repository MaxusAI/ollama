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
| 3, the patch series | **done**: `d96be0a54`. All eight patches apply in order to a fresh b12177 with plain `git apply`. 802, 805 and 903 are re-cut; their changed lines are byte-identical to the v0.40.13 cut (below). 903 compiles for sm_75 and sm_120a, and its guard fires on a mutation in all 10 config tables, which are all of b12177's (`tasks/mmq-compile-0416.sh`). 903's guard-page gate (`tasks/mmq-0416-gate.sh`): running |
| no-GPU harness gates | **green**: `test_verdicts.py`, `test_summarizers.py`, `test_rescore.py` (1 skipped, as in CI), `test_mlx_test_gate.py` |
| 4, image | **not built**: the 8TB array has 36 GB free, and the v0.40.13 fold's builds added 44.6 GB to the `bigdisk` buildkit cache ("Open" below) |
| 5, preflight | not run. The canary reads a run-local expectations copy (`tasks/exp-0416.py`, below) |
| 6, campaigns | not run |
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
- **num_parallel** (84dc81db1): production leaves `OLLAMA_NUM_PARALLEL` at 1, so the removed override changes
  nothing there.

## Canary expectations

The fold's interim builds stamp `0.40.13-dynres-N` (`main`'s last tag is `v0.40.13-dynres`), which `cuda-0-40-13` and
`mlx-cuda-0-40-13` claim, and their `payload_pin` would fail a b12177 build. `tasks/exp-0416.py` writes a run-local
copy that moves the 0-40-13 CUDA pair to `cuda-0-40-16` and `mlx-cuda-0-40-16`, with the b12177 / `54a70511` pins,
908 out of the patchset, and a version pattern admitting the interim stamp. The rows stay 0-40-13's until
`measure_ladder.py` says otherwise. The checked-in profiles come after measurement, as at v0.40.13.

## Open

- **Disk for the image build.** The array has 36 GB free. The `bigdisk` buildkit cache is 82.9 GB: 44.6 GB from the
  v0.40.13 fold's builds (2026-11-03), 32.8 GB from 2026-10-13, and 5.4 GB older. A full build at a new llama.cpp
  and MLX pin adds roughly what the last fold's did.
- 903's guard-page gate on sm_75 and sm_120a.
- Gates 4-6 on CUDA. gfx1151 and Metal are the other hosts'.
