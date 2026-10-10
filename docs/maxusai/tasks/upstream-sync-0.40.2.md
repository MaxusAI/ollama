# TASK: fold upstream v0.40.2 into main

Upstream [v0.40.2](https://github.com/ollama/ollama/releases/tag/v0.40.2) (tag `b061384d9`, 2026-10-08) folds on top
of `main` at `5273e495f` (#459), which carries the v0.35.0 fold.
- **Size:** v0.35.0..v0.40.2 is 38 commits plus 11 merges, 239 files, +31,564/−2,277. That is 106 files added and
  133 modified.
- **Where:** branch `task/upstream-sync-0.40.2`, worktree `.claude/worktrees/upstream-sync-0402` on the Metal host
  (`macbook-pro-m5-max-128GB/mlx-metal`).

**Scope, the maintainer's call (2026-10-10):**
- One fold, straight to v0.40.2, taking v0.35.1, v0.40.0 and v0.40.1 on the way.
- v0.35.1 is not folded on its own. Its pins (llama.cpp b11232, MLX `64ea011c`) are already superseded, so a separate
  0.35.1 fold would measure a payload nobody serves.

## Status (2026-10-10)

| gate | state |
|---|---|
| 1, the merge | **not started**. A trial merge (`git merge-tree`) against `5273e495f` conflicts in 10 files ("Conflicts expected" below) |
| 2, docs and paths | not started |
| 3, the patch series | not started. **Both payloads move** ("Native inputs" below), so every fork patch is re-checked at the new pins |
| no-GPU harness gates | not started |
| 4, image | not started. A full native build: this fold cannot be a Go-only swap |
| 5, preflight | not started. Every profile needs a new measurement, because both payloads move (ADR 0032) |
| 6, campaigns | not started. Each host's call, on its own hardware |
| tag and deploy | not cut, not deployed |

## Native inputs

Both payloads move, so ADR 0032 calls for new profiles, not widened ones.

| input | `main` | v0.35.1 | v0.40.2 |
|---|---|---|---|
| `LLAMA_CPP_VERSION` | b11081 | b11232 | **b11351** |
| `MLX_VERSION` | `59d600b5` | `64ea011c` | **`a59cc231`** |
| `MLX_C_VERSION` | `ebc88f10` | `ebc88f10` | `ebc88f10` |

Upstream's MLX carry patches are now applied through one `cmake/local.cmake` command for MLX and MLX-C.
- `cf6c9de62` carried a Metal residency-refresh patch.
- `cf2a313a2` dropped it again, because `a59cc231` contains it.

## What v0.40.x changes for the fork, as read so far

Each item here is a question for the fold, not yet an answer.

- **Runner-specific manifests in one tag, with MLX preferred on Apple Silicon** (`75b952780`).
  - A tag can now hold one child manifest per runner (`mlx`, `llamacpp`, `ggml`).
  - `runnerPreferences()` (`manifest/manifest.go:844`) orders them `mlx, llamacpp, ggml` on darwin/arm64, and
    `llamacpp, ggml, mlx` everywhere else.
  - So **one tag can resolve to different weights on the Metal host than on the CUDA and ROCm hosts**. Comparing hosts
    by tag, the "q4_K_M/q8 tags are GGUF" convention, and checkpoint identity by manifest digest all need a rule for
    which child was resolved.
- **Background conversion of legacy GGUFs** (`75b952780`, `compatmigrate/`).
  - On first load, a legacy (`ggml`) GGUF is converted into a `llamacpp` child, and the original is kept as a backup.
  - The converter covers every GGUF family the Metal host serves: gemma4, qwen35/qwen35moe, nemotron_h_omni,
    qwen3vl (chandra) and qwen2vl (olmocr2).
  - On that host, the production store holds 14 GGUF model files, 270 GiB in all, with 696 GiB free. A deploy would
    convert each model on its first load, add up to that much on disk, and change the file llama-server loads.
  - GGUF baselines taken on a converted model are new measurements.
- **The MLX tokenizer now matches the original tokenizers' semantics** (`195f4cdca`, +2,180/−495). This covers
  pretokenizer order, split behaviour, Unicode boundaries, added-token normalization and ranked BPE merges. MLX
  token counts may move, and a moved count is a finding to measure, not to wave through.
- **Slow first request after GPU idle** (`cf6c9de62`, upstream #18744). The MLX runner now sets
  `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=1000` unless the environment sets it. Fork campaigns wait for 15 idle
  minutes before measuring, so whether their first cells paid this cost is worth checking. Not yet measured.
- **New MLX surface:**
  - System One and decision models on MLX (`a00d083bb`, `27a36513d`): Laya, Clef/Clef Flash, Tev1, Nimble and
    Strands.
  - Multimodal embeddings with `embeddinggemma-2` (`efe43c556`), which reuses gemma4's vision and audio towers.
- **Outside the runners:** OpenAI tool-result handling (four commits), cloud usage/balance proxying, CLI onboarding,
  and `ollama launch claude` context settings.

## Conflicts expected

A trial merge of v0.40.2 into `5273e495f` (`git merge-tree --write-tree`) conflicts in 10 files. Nine are in
`mlxrunner/`:

- `mlxrunner/client.go`
- `mlxrunner/runner.go`
- `mlxrunner/model/gemma4/`: `audio.go`, `media.go`, `media_test.go`, `process_audio.go`, `process_audio_test.go`,
  `process_image.go`, `vision.go`
- `server/sched_test.go`

The gemma4 files are where the fork's MLX vision work meets upstream's `embeddinggemma-2`, which shares gemma4's
towers. v0.35.1 alone merges into `main` with no conflicts.

**A patch-number clash that merges clean.** Upstream adds `llama/compat/002-clef.patch`, and the fork carries
`llama/compat/002-llama-cpp-nemotron-dynres.patch`. Both would sit at `002`. Gate 3 checks their apply order and
whether they touch the same files.

## Who does what

- **Metal host** (`macbook-pro-m5-max-128GB/mlx-metal`): gates 1–3, then gates 4–6 on Apple Silicon. GPU steps wait
  for production to be idle and for High Power mode, as usual.
- **CUDA and ROCm hosts:** gates 4–6 on their own hardware, when this branch is ready for them. No ask is sent until
  then.
