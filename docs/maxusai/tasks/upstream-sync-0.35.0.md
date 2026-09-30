# TASK: fold upstream v0.35.0 into main

Upstream [v0.35.0](https://github.com/ollama/ollama/releases/tag/v0.35.0) (tag `cc4069396`, 2026-09-28) folds on top of
`main` at `293f221a0` (#426), which carries the v0.34.4 fold.
- **Size:** seven commits, 66 files, +5,599/−998. That is 17 files added, 41 modified and 8 renamed, the renames all
  `x/transfer/` → `transfer/`.
- **Where:** branch `task/upstream-sync-0.35.0`, worktree `claude-scratch/wt-sync0350` on the CUDA host
  (`ai-server/mlx-cuda`).

**Scope, the maintainer's call (2026-09-30):**
- v0.35.0 only, with no GPU0 time: GPU0 is reserved for other work.
- v0.35.1 waits for a proper release. Its rc0 moves llama.cpp to b11232 and MLX to `64ea011c`, where v0.35.0 moves
  neither.

## Status (2026-09-30)

| gate | state |
|---|---|
| 1, the merge | **done**: `83c06ece0`, with one conflicted file (`server/routes.go`, one hunk). `go build ./...` and `go vet` are clean on the touched packages. `go test ./...` passes 58 packages and fails 1 test, a pre-existing load flake (below). gofumpt is clean |
| 2, docs and paths | **nothing to re-point**: the `transfer/` move is upstream's own, and no fork file names `x/transfer`. `check_source_paths.py` is clean |
| 3, the patch series | **not applicable**: llama.cpp `b11081`, MLX `59d600b5` and MLX-C are all identical to `main` |
| no-GPU harness gates | **green**: `test_verdicts.py` 196 OK, `test_summarizers.py`, `test_rescore.py` (1 skipped, the optional corpus test, as in CI), `test_mlx_test_gate.py` |
| 4, image | **not built**. The fold moves no native input, so the release image is a Go-only swap onto a deployed payload ("Gate 4" below) |
| 5, preflight | **not run**. Interim builds resolve on every surface; a `v0.35.0-dynres` tag resolves on none until each host widens its profile ("Preflight profiles" below) |
| 6, campaigns | **CUDA: not run.** GPU0 is reserved, and the gate-4 argument covers the native side. The Go side is upstream's new code plus the fork's delta, carried line for line. gfx1151 and Apple Silicon: each host's call |
| tag and deploy | **not cut, not deployed** |

## What v0.35.0 changes for the fork

- **The System One scoring API** (upstream #18606, `ffb6f220d`). `POST /v1/systemone` works as follows:
  - It compiles typed questions from a new package, `decision/`, and renders them through the model's chat template
    with thinking off.
  - It scores their allowed answers through a new `llm.Scorer`, which posts to llama-server's own `/completion` with
    `n_probs`, and returns probabilities.
  - It accepts only local GGUF models on the qwen3.5 renderer, or in the qwen35 family with no renderer. Nimble and
    Tev are such models. It caps a request body at 64 KiB.

  **One production model passes that gate.** qwen3.6:35b-a3b-q4_K_M's renderer is `qwen3.5`, so a deployed v0.35.0
  would answer System One requests on it. Upstream says callers must choose weights trained for the prompt format.
  qwen3.8 (renderer `qwen3.8`), gemma4 and nemotron3 are refused. Nothing else in production's routes changes.
- **`typical_p` accepted per request again** (upstream #18627, `7af393188`).
  - v0.34.4, which production serves, rejects a request carrying `typical_p` with a 400.
  - v0.35.0 accepts it and logs a deprecation warning (`warnDeprecatedOptions`), and still rejects it as a model
    parameter.

  This relaxes an error that production clients could hit today. No fork tool sends `typical_p`.
- **MLX pulls** (upstream #18625, `cc4069396`). Stalled blob downloads get bounded retries that the watchdog can
  interrupt. `x/transfer` becomes the top-level `transfer/` package.
  - The fork never changed `x/transfer`.
  - The merged `transfer/` equals upstream's, byte for byte.
  - `server/images.go` imports it under the new path. Two of upstream's own comments there still say `x/transfer`.
- **Out of the server image:** benchmark prompts from HumanEval (upstream #17480), and three desktop-app fixes (upstream
  #18598, #18622, #18626).

## The conflict and its resolution

**`server/routes.go`, one hunk, positional.** Both sides added code at the same place, before `EmbedHandler`:
- **The fork:** the think+format helpers (`formatConstrains`, `thinkingContinuationHeadroom`, `reportedPassMetrics`,
  `transitionPassMetrics` and `reclassifyConstrainedThinking`).
- **Upstream:** `SystemOneHandler`.

The resolution keeps both, the fork's first. Upstream's other hunks in the file merged clean:
- the `decision` import;
- the `/v1/systemone` route;
- `warnDeprecatedOptions`;
- the removal of the `typical_p` rejection from `scheduleRunner` and `handleScheduleError`.

None of them touches the fork's code.

**The carry is exact.** The fork's delta is 533 files, +178,425/−3,082 against v0.34.4 before the fold, and the same
against v0.35.0 after it. File by file, its added and removed lines are identical.

## Auto-merges read rather than trusted

The four other files both sides changed:

| file | ours | upstream | why it holds |
|---|---|---|---|
| `api/types.go` | `ImageMin/MaxTokens`, `KVCacheType` and their defaults (+33) | `TypicalP`'s deprecation comment (one line) | disjoint |
| `docs/api.md` | the fork's options text (+3/−1) | the `typical_p` deprecation note (+2) | disjoint paragraphs |
| `server/images.go` | gemma4 safetensors suppresses audio (ADR 0021) (+7) | the `transfer` import path | disjoint |
| `server/routes_generate_test.go` | the ADR 0004/0045 flow tests (+1,986/−43) | the `typical_p` rejection tests become "deprecated option passes through" | disjoint; `server` passes |

## Gate 4: a Go-only swap is valid here, minus one inert patch

This fold changes no native input. Against `main`, nothing moves under any of these:
- `LLAMA_CPP_VERSION`, `MLX_VERSION` or `MLX_C_VERSION`;
- `llama/`, `ml/`, `runner/`, `mlx/` or `mlxrunner/`;
- `cmake/`, `CMakeLists.txt`, `CMakePresets.json` or `Dockerfile`.

Upstream's diff has no C, C++, CUDA, Metal or patch files. Its one Objective-C file, `app/cmd/app/app_darwin.m`,
belongs to the macOS desktop app.

**`main` has moved since the deployed payloads were built.** `llama/compat/805-fattn-force-tile.patch`
(`e68e658c0`, ADR 0041) entered `main` with #371 (`c8e8ecc80`, 2026-09-29). That was after `b43ee8e37`, which every
host deployed.
- 805 is the only native difference, with its README entry.
- It is gated on `GGML_CUDA_FATTN_FORCE_TILE`. Unset, it changes nothing: flash attention picks its kernel as before.
- So a Go-only swap onto a deployed payload serves what a full build of this tree would serve. It lacks only a switch
  nobody has set.
- A build that needs 805 must be a full build.
- 805's own comment still calls it "experiment, not for shipping". Changing that comment changes patched source, so it
  waits for the next native build: v0.35.1's.

## Preflight profiles

These are resolved with the harness's own `resolve_profile`:

| surface | interim build (`0.34.4-dynres-N-g…`) | after a `v0.35.0-dynres` tag (`0.35.0-dynres-0-g…`) |
|---|---|---|
| cuda | `cuda-dynres-903` | **none**: the pattern admits `0.3[234]` |
| mlx-cuda | `mlx-cuda` | **none**: the same pattern |
| rocm7 | `rocm7-0-34-4-dynres` | **none**: it admits `0.34.[34]` |
| mlx-metal | `mlx-metal-0-34-4` | **none**: the same |

The payload does not move, so ADR 0032 says to widen each profile rather than cut a new one, as the 0.33.0 and 0.34.0
folds did for CUDA. No profile is edited here. Each host widens its own after its own run. On CUDA that waits until
GPU0 is free.

## The one failing test

`cmd/launch`'s `TestCodexAppCountsOnlyOllamaRequestsInRegularProfile` fails under this host's load (load average about
20 on 32 cores): "regular profile Ollama request count = 0, want 2". It is not the fold's:
- Neither the fork nor v0.35.0 touches `cmd/launch`.
- Interleaved, 10 runs at a time and each batch in a fresh container, it failed 36 of 40 runs on `main` and 27 of 40
  on the fold.
- CI's `test` workflow has passed on every recent PR.
