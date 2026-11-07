# TASK: the Metal leg of v0.40.13-dynres (Apple Silicon, mlx-metal and metal)

Claimed by the Metal host (`macbook-pro-m5-max-128GB/mlx-metal`) on 2026-11-07, at the maintainer's word. Status:
**running.**

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

`macbook-pro-m5-max-128GB/mlx-metal`
