# TASK: fold upstream v0.40.5 into main

Upstream [v0.40.5](https://github.com/ollama/ollama/releases/tag/v0.40.5) (tag `d919272e5`, 2026-10-14) folds on top
of `main` at `a4faff320` (#497), which carries the v0.40.3 fold (tag `v0.40.3-dynres`).
- **Size:** v0.40.3..v0.40.5 is 51 commits, 172 files, +8,053/−3,831. It takes v0.40.4 on the way.
- **Where:** branch `task/upstream-sync-0.40.5`, on the CUDA host (`ai-server/mlx-cuda`).
- **Why now:** v0.40.5 adds NVIDIA Nemotron 3.5 Super (120B-A12B, a hybrid Mamba-2/Transformer MoE with a C-RADIO
  vision encoder) and Nemotron 3.5 Bolt (74B-A9B, with a drafter). The library gates both on the client version:
  on `0.40.2-dynres.1-17-g75c6d35` a pull answers "requires ollama version v0.40.5 or newer" (Super) and HTTP 412
  (Bolt). The architecture needs v0.40.5's llama.cpp and MLX, so fetching the blobs by hand would not help.

## Status (2026-10-16)

| gate | state |
|---|---|
| 1, the merge | **done**: `bd36fd015`, 5 conflicted files ("The conflicts and their resolution" below). `go build` and `go vet` are clean; `go test` passes 63 packages. The one failure, `cmd/launch` `TestCodexAppCountsOnlyOllamaRequestsInRegularProfile`, is an upstream flake: it also fails 1 of 3 runs on a clean v0.40.5 checkout |
| 2, docs and paths | **clean**: `check_source_paths.py --changed-since origin/main` resolves every reference |
| 3, the patch series | **done**: `8761533a6`. All nine patches apply in order to a clean b11669. 002 is re-anchored and 903 is cut to guard-only; 903's retirement gate passes on sm_120a and sm_75 ("Gate 3" below) |
| no-GPU harness gates | **green**: `test_verdicts.py`, `test_summarizers.py`, `test_rescore.py` (1 skipped, as in CI), `test_mlx_test_gate.py` |
| 4, image | **building on CUDA**: `maxusai/ollama:sync-0.40.5-fold`, stamp `0.40.3-dynres-12-g8761533`. A full native build: all three pins move |
| 5, preflight | not run. Every profile needs a new measurement, because the payloads move (ADR 0032) |
| 6, campaigns | not run |
| tag and deploy | not cut, not deployed |

## Native inputs

| input | `main` (v0.40.3 fold) | v0.40.5 |
|---|---|---|
| `LLAMA_CPP_VERSION` | b11351 | **b11669** |
| `MLX_VERSION` | `a59cc231` | **`bebd6422`** |
| `MLX_C_VERSION` | `ebc88f10` | **`a341b492`** |

- **llama.cpp b11669** brings native decision-model (clef) support, so upstream drops its `llama/compat/002-clef.patch`
  and `llama/clef/`. It also adds Nemotron Super 3.5 VL and Bolt 74B (#30520), a separate `PROJECTOR_TYPE_NEMOTRON_OMNI`
  with native dynamic resolution, and llama.cpp#29953's MMQ fix (`fc9ce6b9d`).
- **MLX `bebd6422`** comes with Go kernel wrappers in `mlx/` for the Nemotron 3.5 path: `depthwise_conv.go` and
  `weighted_sum.go` are new, `mamba2_scan.go` and `gated_delta.go` extended.

## The conflicts and their resolution

- **`mlxrunner/model/gemma4/gemma4.go`.** Upstream #18809 drops the manual CUDA prefill attention for
  `head_dim > 128` and runs MLX SDPA everywhere. Taken, with the fork's bidirectional vision-chunk mask in place of
  upstream's `baseMask` parameter. **This changes gemma4 prefill numerics on CUDA MLX**, so gemma4's mlx-cuda cells
  are new measurements.
- **`mlxrunner/model/nemotron_h/vision_prompt.go`.** Upstream's `nemotronImagePatchGrid` now returns an error. The
  fork keeps its per-request floor (`withMinNumPatches`); `media_budget_test.go` follows the new signature.
- **`mlxrunner/pipeline.go`.** Upstream #18886 and the fork's `guardClose` fix the same bug, a cleanup panic
  masking the first one, in two ways. Upstream skips stateful cleanup while a panic unwinds, so a failed request
  saves no partial cache; the fork ran the cleanup under its own recover. Upstream's `skipCleanupOnPanic` is taken
  at every site, and now re-raises the panic wrapped once as the fork's `*firstPanic` (`asFirstPanic` in
  `unwind.go`), so the first panic's stack still reaches the log and the client. `pipeline_panic_test.go` unwraps it
  before comparing. `guardClose` stays for the pipeline teardown.
- **`server/sched.go`.** Upstream folds the generation-batch surcharge into `llm.LlamaServerBatch`, with the same
  768 MiB and 2 GiB steps plus whole-prompt decision models. The fork's image-chunk floor on
  `automaticGenerationBatch` stays.
- **`server/routes.go`.** Upstream #18947 reuses loaded models across requests, so it stops writing
  `m.Config.Parser = "harmony"` into the model and names the parser with `parserNameForModel`. Applied to both of
  the fork's generate-handler parser blocks (the fork splits upstream's block by thinking mode) and to its
  chat-handler trace logs. Upstream's `systemOneLlamaServer` is added beside the fork's structured-outputs helpers.

## Gate 3: the patch series at b11669

Applied in the applier's order to a clean b11669 with plain `git apply`:

| patch | result |
|---|---|
| 001, 004, 005, 801, 802, 805, 908 | apply unchanged |
| 002 nemotron dynres | **re-anchored**. b11669's new `NEMOTRON_OMNI` case sits right after `NEMOTRON_V2_VL` in `clip.cpp` and `mtmd.cpp`, which moved both hunks' trailing context. The changed lines are identical, +20/−3 |
| 903 MMQ padding | **guard-only**. The pin contains #29953 itself, so the backported padding goes; the y-tile helper and its `static_assert`s stay, re-anchored on #30168's template line |
| `mlx/compat/mlx-c` 0001, 0002 | upstream's own, unchanged; apply to MLX-C `a341b492` |

- **002 is still needed.** The compat layer, upstream's included (`llama-ollama-compat.cpp`), maps the
  `nemotron_h_omni` GGUFs that nemotron3 ships to the `nemotron_v2_vl` projector. b11669 still preprocesses that
  projector on a fixed canvas. Upstream's new `NEMOTRON_OMNI` path serves natively converted GGUFs.
- **903's padding retires; the gate passes on CUDA.** The gate is "For the fork" in
  `docs/maxusai/upstream-mmq-29953-material.md`:
  - Step 1: the series applies without the padding.
  - Step 3, on b11669 plus the fold's 805, 903 and 908 (`tasks/mmq-0405-gate.sh`, results in
    `tasks/mmq-successor-results/gate-0405-sm*.tsv`):
    - The tree passes every case in every mode on both GPUs: 3/3 under each guard page and on the stock pool, and 0
      memcheck errors at exact allocation.
    - The `J_best`-block src1 control faults 3/3 under `guard:src1` on `ids16` (sm_120a and sm_75) and on
      `dense321` (sm_75).
    - The no-padding `ids_dst` control faults 3/3 under `guard:ids_dst` on every MoE case.
    - `j100_b0` and `j100_b1`'s src1 control faults 1 of 3 on sm_120a. That is a rate, so those rows are not
      evidence there, as on #462.
  - The guard's `static_assert`s compile in the gate-4 image build (nvcc); hipcc is the ROCm host's.
- **Read, not yet measured:**
  - 908 applies around upstream's FA race fix (#30103) and whole-tile scheduling (#29435). Its loop-rate gate on
    gemma4:26b think-on must be re-measured.
  - `clip_graph::build_attn` now asserts an f16 mask instead of casting one. No fork patch and no
    `nemotron_v2_vl` graph passes a mask.
  - `mtmd-image.cpp`'s changes touch the llava-uhd and longest-edge preprocessors only, not `dyn_size`.

## Open

- Gate 4 on CUDA, then gate 5 against new `-0-40-5` profiles on each platform, each host's own gate 6.
- 908's loop-rate gate on b11669; hipcc for 903's guard (the ROCm host).
- The Nemotron 3.5 models themselves: pull, load and a vision check on each engine they ship for.
