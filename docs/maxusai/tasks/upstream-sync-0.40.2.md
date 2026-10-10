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

## Status (2026-10-10, 22:55 AEST)

| gate | state |
|---|---|
| 1, the merge | **done**: `b6f672df2`, 10 conflicted files ("The conflicts and their resolution" below). `go build` and `go vet` are clean on all 85 non-app packages, and `go test` passes the 52 that need no GPU (38 with tests). The fork's delta carries exactly, apart from the resolved files and one gofmt realignment |
| 2, docs and paths | **clean**: `check_source_paths.py --changed-since origin/main` resolves every reference. The full-tree run's 7 unresolved references are already on `main` |
| 3, the patch series | **done**: `948ef3aab`. At b11351 the series applied through 802; 805 and 903 failed and are re-cut ("Gate 3" below). All ten patches apply in the applier's order to a clean b11351, and in the real build |
| no-GPU harness gates | **green**: `test_verdicts.py` 238 OK, `test_summarizers.py`, `test_rescore.py` (1 skipped, as in CI), `test_mlx_test_gate.py` |
| 4, image | **built on Metal**: `0.35.0-dynres-27-g948ef3a` (an interim stamp: no `v0.40.2-dynres` tag yet), with llama-server at `631109b34` (b11351). `CLEAN_DEPS=1`, 3 min 40 s. A full native build: this fold cannot be a Go-only swap |
| 5, preflight | **Metal: PASS=23 SKIP=16** on the interim build, against the new `mlx-metal-0-40-2` ("Gate 5 on Apple Silicon" below). CUDA and ROCm: not run. Every profile needs a new measurement, because both payloads move (ADR 0032) |
| 6, campaigns | **Metal: running** (the seven think-off cells and OCRBench rows 0–199, against production's own v0.35.0 runs). The native gate, the GGUF conversion and rollback, and the smoke all pass. CUDA and ROCm: each host's call |
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
- **System One now requires a decision model** (v0.35.1: #18708, #18737). `/v1/systemone` accepts only a model
  that declares the `decision` capability (GGUF `decision.type`, or an explicit capability at create). Production's
  0.35.0 answers System One on qwen3.6:35b-a3b-q4_K_M, the one model on the qwen3.5 renderer. On this fold it
  answers 400, "does not support decision", as upstream intends. No fork tool calls System One.
- **Outside the runners:** OpenAI tool-result handling (four commits), cloud usage/balance proxying, CLI onboarding,
  and `ollama launch claude` context settings.

## The conflicts and their resolution

The merge of v0.40.2 into `5273e495f` conflicted in 10 files, nine of them in `mlxrunner/`. v0.35.1 alone would
have merged with none.

- **`mlxrunner/runner.go`: both sides.** The fork's "report first, stop second" exit on a `fatalRunnerError` stays
  in the completion case, and upstream's new embedding case follows it. `runEmbed` never returns a
  `fatalRunnerError`, so the embedding case only logs, as upstream wrote it.
- **`mlxrunner/client.go`: both sides.** The fork's `Client` fields (the atomic `softContextLength`, `numCtxAuto`,
  `kvEstimate`) plus upstream's `embeddingDimensions`.
- **`server/sched_test.go`: the fork's form.** The fork runs this test under `synctest`, with plain receives and no
  context timeout. Upstream's change moves `b.ctxDone()` after the VRAM report; the resolution takes that move into
  the fork's form.
- **`mlxrunner/model/gemma4/`, seven files: the fork's pipeline, as in the v0.34.4 fold.** That fold kept upstream's
  per-image resolution out (ADR 0003, 0008, 0021), and upstream's `audio.go`, `process_audio.go`,
  `process_image.go` and their tests stayed deleted. They stay deleted here, along with two new upstream tests of
  that API (`multimodal_export_test.go`, `process_image_test.go`). `gemma4.go` auto-merged and takes upstream's
  `rope_type` check.

### embeddinggemma-2 is not in this fold

Upstream's multimodal embedding model (#18820, `efe43c556`) is built on gemma4's upstream towers. It calls
`gemma4.LoadVisionTower`, `ProcessImage`, `ImageGeometry`, `LoadAudioTower` and `ProcessAudio`. None of those
exists in the fork, whose gemma4 vision code is its own (`VisionTower.Forward` over patches, the budget-fill
pipeline). Taking it would mean one of two things:
- reversing the v0.34.4 decision on gemma4's image pipeline, which needs an ADR with a measurement;
- carrying upstream's towers a second time, beside the fork's, for one model.

So `mlxrunner/model/gemma4embedding/` is removed and unregistered. The generic `/api/embed` media path,
`mlxrunner/embed.go` and `create/gemma4embedding.go` stay. On this fold, an `embeddinggemma-2` model fails to load
with an unknown architecture. No host serves it. **Deferred**, with the gemma4 pipeline question it depends on.

### The carry

The fork's delta against v0.35.0 before the fold, compared line for line with its delta against v0.40.2 after it,
differs only in the resolved files above and in one gofmt realignment of a map literal in
`server/routes_generate_test.go`.

## Gate 3: the patch series at b11351

Applied in the applier's order (`cmake/apply-git-patches.cmake`, sorted by name) to a clean b11351 checkout:

| patch | at b11351 |
|---|---|
| 001 hooks, 002 nemotron dynres, 004, 005, 801, 802, 908 | apply as they are |
| 002-clef (upstream's, new) | applies, before the fork's 002. The two touch different files |
| 805 fattn force-tile | **re-cut, context only**. Upstream templated `flash_attn_mask_to_sparse_indices` and moved the kernel selector down 12 lines. The added and removed lines are identical to the b11081 cut |
| 903 MMQ padding | **re-cut, one semantic change**, below |

**903 is still needed.** b11351 still sizes both src1 allocations with `ggml_cuda_mmq_get_J_max()`.

One hunk failed on context, and b11351 changes one thing 903 depends on: `ggml_cuda_mmq_get_config()` gains a
`prec_src1` argument. On Blackwell, NVFP4 and MXFP4 now use the ampere table unless src1 is Q4. The re-cut
computes `J_pad` after `prec_src1` and passes it, so the padding comes from the table the launch uses, not the
default-precision one. Otherwise its added and removed lines match the b11081 cut.

The compile-time guard's table functions keep their `(type, J, fallback)` signature at b11351. **Not compiled
here**: the Metal host has no nvcc or hipcc, so the CUDA and ROCm hosts' builds are where the guard runs.

## On Apple Silicon

Every GPU step ran on the fold's own stage: `~/.ollama/binaries/stage-fold-0402`, holding its binary and its
`lib/ollama`. It ran under production's gate, which means production idle for 15 minutes and powermode 2, with a
step cancelled and re-run if production woke.

**The store is an APFS clone, `~/.ollama/models-mlx-fold0402` (`cp -cR`), not production's.** v0.40.2 converts a
legacy GGUF on its first load and writes the result into the store. Against production's own store, validation
would have rewritten production's manifests while 0.35.0 served them. The clone cost no disk until the conversion
wrote.

### The native gate

`go test ./mlx/... ./mlxrunner/... -p 1` with `OLLAMA_VISION_E2E=1`, on the fold's payload: **1,208 passed,
0 failed, 9 skipped**, 25 packages. `TestVisionGoldenParity` and `TestVisionEndToEnd` pass on real weights. The
0.34.4 and 0.35.0 runs passed 806, but they ran `./mlxrunner/...` alone, so the counts are not comparable. This run
adds `./mlx/...`, plus upstream's new System One, Laya, Clef and Strands tests.

### GGUF conversion, and the rollback it has to survive

One 1-token request per model on the fold's stage, then on production's deployed 0.35.0 build
(`0.35.0-dynres-0-g043f441`) against the same converted clone:

| model | converted in | the fold then loads | 0.35.0 afterwards |
|---|---|---|---|
| gemma4:31b-it-q4_K_M | 24 s | `sha256-3bf4a777…`, 17.4 GiB, new | answers, from the original `sha256-280af683…` |
| qwen3.6:35b-a3b-q4_K_M | 19 s | `sha256-3f87bb0b…`, 21.5 GiB, new | answers, from the original `sha256-f5ee307a…` |

- **Disk.** Each converted model is a second full copy: 17.4 + 21.5 GiB of model blobs, plus 1.9 GiB of other new
  blobs. The 14 GGUF files in production's store hold 270 GiB, and a deploy would convert each one on its first
  load. The original stays as the downgrade anchor.
- **The anchor is how rollback works.** The legacy manifest stays where it was and gains three
  `manifest.v2+json` layers. 0.35.0 reads it and loads the original blob.
- **One rollback wart, cosmetic.** 0.35.0 lists each converted child as its own model,
  `llamacpp:<digest>`: 29 models on the converted clone against production's 27. v0.40.2 hides them (#18874).

### Gate 5 on Apple Silicon

**`mlx-metal-0-40-2` is a new profile, measured on this payload.**
- The four ladders were measured with `measure_ladder.py`, one fresh stage per arch, and are byte-identical to 0-34-4's.
- Every engine init reported MLX `0.32.3-53-ga59cc23`, and llama-server reports `631109b34`.
- Its pattern admits only the `v0.40.2-dynres` tag's lineage.

**The interim build was run against a campaign copy of the file.** The interim build stamps `0.35.0-dynres-27-g948ef3a`,
which falls in 0-34-4's lineage. The campaign copy's only change pinned 0-40-2's pattern to that one stamp. Parsed and
compared field by field, once, before the run, the copy and the branch file agree everywhere else.

Preflight `--quality` on the stage: **PASS=23 SKIP=16**, the same shape as the v0.35.0 release run. The 16 skips
are the native path's usual ones: no container, no load_hparams line, and no quality, pinned-budget or poison
expectation recorded for these arches.

### The smoke

| request | result |
|---|---|
| think+format, two-pass, on MLX (`gemma4:26b-nvfp4`) | pass one drafted, 110 of 138 accepted; content parses as JSON |
| think+format, two-pass, on the converted GGUF (`gemma4:31b-it-q4_K_M`) | content parses as JSON; llama-server loads the converted blob |
| an image on the converted GGUF | `prompt_eval_count` 573 and a correct description, so the projector survived the conversion |
| System One on `qwen3.6:35b-a3b-q4_K_M` | 400, "does not support decision", as upstream intends since v0.35.1 |

## Who does what

- **Metal host** (`macbook-pro-m5-max-128GB/mlx-metal`): gates 1–3, then gates 4–6 on Apple Silicon. GPU steps wait
  for production to be idle and for High Power mode, as usual.
- **CUDA and ROCm hosts:** gates 4–6 on their own hardware, when this branch is ready for them. No ask is sent until
  then.
