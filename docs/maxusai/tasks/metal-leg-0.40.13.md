# TASK: the Metal leg of v0.40.13-dynres (Apple Silicon, mlx-metal and metal)

Claimed by the Metal host (`macbook-pro-m5-max-128GB/mlx-metal`) on 2026-11-07, at the maintainer's word. Status:
**done** (2026-11-08). Apple Silicon serves `0.40.13-dynres-0-g85d6368` on `:11435` since 2026-11-08 11:16.

The v0.40.5 and v0.40.13 folds (#498, #501) merged, and `v0.40.13-dynres` (`85d636818`) is tagged. The CUDA and ROCm
legs are done. Apple Silicon still serves `0.40.2-dynres.1-17-g75c6d35`: the v0.40.3 fold's Go on the v0.40.2 payload
(MLX `a59cc231`, MLX-C `ebc88f10`, llama.cpp b11351). This leg builds the tag natively, measures the two Metal
profiles the payload move needs, runs gate 6, and deploys the tag on Apple Silicon.

## Scope

- **Metal gates the tag only.** It writes no `-0-40-5` profiles: Apple Silicon never serves b11669 or MLX `bebd6422`,
  and the tag contains all of #498. That's the precedent the v0.40.2 fold set when it skipped v0.35.1. A build of
  #498 stays the fallback, only if a moved cell needs attributing between the two pins.
- **Not repeated here:** the merges, their MLX conflicts, the 004/903 re-cuts and 908's retirement. Those are the CUDA
  host's (#498, #501, #505), and 903's hipcc gate is the ROCm host's.
- **Only Metal can do these:**
  - the native macOS build, which compiles upstream's new `002-metal-unary-cast-constexpr`;
  - the MLX Go tests against a real libmlx on these merges, which the CUDA and ROCm runs skipped;
  - the `mlx-metal-0-40-13` and `metal-0-40-13` profiles;
  - the Metal kernel numerics the new pins move: MLX fp4 dequant, split-K partials, the verify matmul and attention
    kernels, the fused sliding-window prefill, and llama.cpp's M5 tensor-API flash attention.

## Gates

| # | gate | power |
|---|---|---|
| 1 | Metal toolchain probe, disk, pinned python | — |
| 2 | native build at the tag in a detached worktree (`CLEAN_DEPS=1`), payload proof | — |
| 3 | `go test ./mlx/... ./mlxrunner/... -p 1`, `OLLAMA_VISION_E2E=1` | any |
| 4 | the APFS clone of production's store; rollback check to the outgoing build | any |
| 5 | ladders for `mlx-metal-0-40-13` (gemma4, gemma4_unified, qwen35moe, qwen35, muse-glimmer) and `metal-0-40-13` (muse-glimmer plus gemma4 and qwen GGUF rows) | any |
| 6 | preflight `--quality` on both profiles; deploy smoke; a Nemotron 3.5 Super smoke | any |
| 7 | the A/B: seven cells in both think modes, plus OCRBench v1 rows 0–199. A is production's build, B is the tag, both on the same clone and in the same power mode | low (overnight) |
| 8 | the drafting knob's retirement gate (`leak-repro5.sh` with drafting on), for upstream `a5b659e61` (v0.40.10) | any |
| 9 | high-power re-runs for anything whose comparison depends on speed (drafting depth is wall-clock driven) | high (daytime) |
| 10 | the release build at the tag and a stopped swap on `:11435`; preflight on production | — |

**Quality over speed, for now.** The maintainer keeps the host on low power overnight. Each A/B pair runs in one power
mode, and the gate holds rather than mix them. The runs record their power mode.


## Results

**Build.** Both halves at the tag in a detached worktree (`CLEAN_DEPS=1`), stamp `0.40.13-dynres-0-g85d6368`, MLX
`065ec468`, llama.cpp b11963. The Xcode 27.2 upgrade had removed the Metal toolchain again
(`xcodebuild -downloadComponent MetalToolchain`, run by the maintainer); the probe then read `__METAL_VERSION__` 400.

| gate | result |
|---|---|
| 3, MLX Go tests on a real libmlx | 1659 passed, 0 failed, 10 skipped |
| 5, ladders | every 0-40-2 row re-measured byte-identical; the new GGUF rows (gemma4, qwen35) equal `cuda-0-40-13`'s |
| 6, preflight, stage | `mlx-metal-0-40-13` PASS=26 SKIP=20, `metal-0-40-13` PASS=10 SKIP=22; no `failed to build` line (002-metal compiles) |
| 6, smoke | 4/4. Upstream's `adc4aae9d` rewords System One's refusal ("is not a llama.cpp decision model"), still HTTP 400; the check accepts both |
| 6, Nemotron 3.5 Super | `nemotron-3.5-super:120b-a12b-nvfp4` loads at 74.9 GiB (all on the GPU), answers text and an image, both `stop` |
| 7, think-off A/B, seven cells | below |
| 7, OCRBench v1 rows 0–199 | gemma4:31b-nvfp4 174/200 on both, gemma4:31b-it-q4_K_M 171/200 on both; the same items right and wrong (McNemar p = 1.000) |
| 7, think-on | **not run** for the A/B, at the maintainer's word (2026-11-08). Arm A alone finished gemma4:26b (26/27; `bbox_contract_adv_real` not converged at 131072) and gemma4:31b (27/27 at 16384) |
| 8, drafting knob | **inconclusive.** The size ladder (no format, and with a grammar at `DRAFT_UNDER_GRAMMAR=1`) logged no `speculative decode stats` line on either build: these 5–30-token stop-terminated answers end before qwen3.6's drafter engages on Metal. `held − trie` +0.06 GiB over 15 requests on both. The retention was measured on MLX-CUDA; the knob stays |
| 10, deploy | stopped swap of both halves, 2026-11-08 11:16; production preflight `mlx-metal-0-40-13` 26/20, `metal-0-40-13` 10/22; plist variables unchanged |

**The think-off A/B** (`drv/cmp_thinkoff.py`; A is `0.40.2-dynres.1-17-g75c6d35`, the build production served, B the
tag; same store clone, same suite tree, powermode 2 for both):

| cell | answers identical | A: contract, hits, mean IoU | B: contract, hits, mean IoU |
|---|---|---|---|
| gemma4:12b-nvfp4 | 13/28 | 16/20, 96/120, 0.767 | 16/20, 96/120, 0.769 |
| gemma4:26b-nvfp4 | 26/28 | 12/20, 72/120, 0.583 | **13/20, 78/120, 0.623** |
| gemma4:31b-nvfp4 | 28/28 | 19/20, 114/120, 0.886 | the same |
| qwen3.6:35b-a3b-nvfp4 | 2/28 | 14/20, 92/120, 0.703 | 14/20, 92/120, 0.701 |
| qwen3.8:27b-nvfp4 | 23/28 | 19/20, 114/120, 0.937 | the same |
| gemma4:31b-it-q4_K_M (GGUF) | 3/28 | 16/20, 97/120, 0.738 | 16/20, 97/120, 0.738 |
| qwen3.6:35b-a3b-q4_K_M (GGUF) | 2/28 | 14/20, 89/120, 0.691 | 14/20, 89/120, 0.689 |

Valid JSON 189/189 on both. No cell lost a contract or a hit; gemma4:26b's `bboxm_pin_noanc_pos` went from 0 to 6/6
(IoU 0.042 → 0.855). Answers moved where the pins predicted: the MLX fp4 / split-K / verify kernels, and on GGUF the
gemma4 tower's GELU switch and the Qwen-VL resize.

**Power mode.** On AC this host's profile is powermode 2; the overnight plan assumed 1. Both arms ran at 2; the
gemma4:12b think-off cases that ran at 1 and inside the build window were archived and re-run.

`macbook-pro-m5-max-128GB/mlx-metal`
