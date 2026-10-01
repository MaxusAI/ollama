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
| 4, image | **CUDA: not built**. The fold moves no native input, so the release image is a Go-only swap onto a deployed payload ("Gate 4" below). **gfx1151: built in full** (2026-09-30), `maxusai-ollama:0.34.4-dynres-41-gfedbe05-rocm7-gfx1151`. Its GPU code is production's, byte for byte, and 805 adds 64 bytes of host code ([Gates 4–6 on gfx1151](#gates-46-on-gfx1151-2026-09-30)). **Apple Silicon: Go-only** (2026-10-01), `0.35.0-dynres-0-g043f441` at the tag, on production's payload byte for byte ([Gates 4–6 on Apple Silicon](#gates-46-on-apple-silicon-2026-10-01)) |
| 5, preflight | **CUDA: not run**. Interim builds resolve on every surface; a `v0.35.0-dynres` tag resolves on none until each host widens its profile ("Preflight profiles" below). **gfx1151: PASS=27 SKIP=5** on the fold image, equal to production's own run in every measured value. `rocm7-0-34-4-dynres` now admits `0.35.0`. **Apple Silicon: PASS=23 SKIP=16** on the release build, equal to the deployed build's run in every measured value except two drafted `think_format` rows, which vary between production's own runs. `mlx-metal-0-34-4` now admits `0.35.0` |
| 6, campaigns | **CUDA: not run.** GPU0 is reserved, and the gate-4 argument covers the native side. The Go side is upstream's new code plus the fork's delta, carried line for line. **gfx1151: done** (2026-10-01). Think off, all 4,902 scored cells, all 140 answers and all 200 OCRBench answers equal production's, byte for byte. Think on, no model loses a finish, and qwen3.6 and gemma4:26b are byte-identical. The other differences are sampling draws or the two-pass flow's slot race, which production's own image shows. **Apple Silicon: done** (2026-10-01). Think off, all 196 answers and all 200 OCRBench answers equal the deployed build's, byte for byte. Think on did not run: on Metal, gemma4's long drafted thinking does not repeat byte for byte |
| tag and deploy | **tagged** `v0.35.0-dynres` on `043f441a7` (2026-10-01). gfx1151 deployed it the same day ([#427](https://github.com/MaxusAI/ollama/pull/427#issuecomment-5930629034)). Apple Silicon: validated, not deployed |

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
| rocm7 | `rocm7-0-34-4-dynres` | `rocm7-0-34-4-dynres`, widened by gfx1151 after its run (2026-09-30) |
| mlx-metal | `mlx-metal-0-34-4` | `mlx-metal-0-34-4`, widened by the Metal host for its run (2026-10-01) |

The payload does not move, so ADR 0032 says to widen each profile rather than cut a new one, as the 0.33.0 and 0.34.0
folds did for CUDA. The fold's own commits edit no profile. Each host widens its own after its own run:
- gfx1151 widened `rocm7-0-34-4-dynres` on 2026-09-30 ([The rocm7 profile, widened](#the-rocm7-profile-widened)).
- The Metal host widened `mlx-metal-0-34-4` on 2026-10-01 ([The mlx-metal profile, widened](#the-mlx-metal-profile-widened)).
- On CUDA that waits until GPU0 is free.

## The one failing test

`cmd/launch`'s `TestCodexAppCountsOnlyOllamaRequestsInRegularProfile` fails under this host's load (load average about
20 on 32 cores): "regular profile Ollama request count = 0, want 2". It is not the fold's:
- Neither the fork nor v0.35.0 touches `cmd/launch`.
- Interleaved, 10 runs at a time and each batch in a fresh container, it failed 36 of 40 runs on `main` and 27 of 40
  on the fold.
- CI's `test` workflow has passed on every recent PR.

## Gates 4–6 on gfx1151 (2026-09-30)

**Host.** The ROCm host, `amd-server`: Ryzen AI Max+ 395 with a Radeon 8060S (gfx1151), 96 GiB of VRAM.
- The maintainer started this leg on 2026-09-30. It ran from 20:19 AEST that evening to 13:27 on 2026-10-01.
- Every gate ran in a bench container on `127.0.0.1:11498`, in production's environment: an f16 KV cache, flash
  attention on, the two-pass flow (`OLLAMA_FORMAT_TWO_PASS=1`) and two parallel slots.
- Production on `:11434` kept serving 0.34.4 and shared the GPU. So did an idle ROCm 10 probe container, which also
  ran during the baseline.
- **The reference is production's own build,** `0.34.4-dynres-0-gb43ee8e`
  (`maxusai-ollama:0.34.4-rocm724-main-b43ee8e3`), measured in the same environment. Its preflight run is #415's.
  Its campaign is the gfx1151 gate's 2026-09-28 baseline, `r0344base_`
  ([amd-upgrade-gate.md](../amd-upgrade-gate.md#2026-09-28s-baseline-productions-configuration-measured)).
- The scripts and tools named below are in the run directory. `fold-0350.sh` ran the gates, and `cold0350.sh` the
  cold captures.

**On gfx1151 the fold changes no result that these gates measure.**
- Its GPU code is production's, byte for byte. 805 adds 64 bytes of host code.
- Its preflight repeats production's run in 30 of 32 rows, every measured value included. The other two rows name
  the build.
- With thinking off, all 4,902 scored cells and all 140 answers equal production's, byte for byte. So do all 200
  OCRBench answers.
- With thinking on, no model loses a finish. qwen3.6 and gemma4:26b are byte-identical, their loops included. Every
  other difference is a sampling draw, or comes from the two-pass flow's slot race, which production's own image
  shows.
- System One and per-request `typical_p` work as upstream describes, and `typical_p` reaches llama-server's sampler.

### Gate 4: the image

| | the fold | production |
|---|---|---|
| image | `maxusai-ollama:0.34.4-dynres-41-gfedbe05-rocm7-gfx1151` | `maxusai-ollama:0.34.4-rocm724-main-b43ee8e3` |
| built from | `fedbe0507`, the fold's head, through `scripts/build_rocm.sh` (`ROCM_TOOLCHAIN=rocm7`, `AMDGPU_TARGETS=gfx1151`): `Dockerfile.rocm` on `rocm/dev-ubuntu-24.04:7.2.4-complete` | `b43ee8e37`, the same way |
| payload | 1863 entries, 96 gfx1151 rocBLAS kernel files | 1863 and 96. No file on one side only, and no SONAME or symlink change |

- **It is a full build, not a Go-only swap.** It took 26 s from ccache. The ROCm build applies every patch in
  `llama/compat/`, so the image carries 805.
- **One of the five native libraries differs: `libggml-hip.so`.** `llama-server`, `libllama-server-impl.so`,
  `libggml-base.so` and `libmtmd.so` have production's sha256.
- **That library's GPU code is byte-identical.** Its `.hip_fatbin` section holds every gfx1151 code object.
  `fatbin.txt`, verbatim:

```
prod: libggml-hip.so 74064600 bytes; .hip_fatbin 60336816 bytes, sha256 8ec4a0d4c48b1774; GGML_CUDA_FATTN_FORCE_TILE strings: 0
fold: libggml-hip.so 74064664 bytes; .hip_fatbin 60336816 bytes, sha256 8ec4a0d4c48b1774; GGML_CUDA_FATTN_FORCE_TILE strings: 1
.hip_fatbin: byte-identical
sections whose size differs:
< .text b2ac04
> .text b2ac44
< .relro_padding 000ef8
> .relro_padding 000eb8
```

- The host code is 64 bytes longer, and the library has one more string, `GGML_CUDA_FATTN_FORCE_TILE`. That is 805's
  switch. Neither the bench container nor production sets it, so flash attention picks its kernels as before.
- So on gfx1151, "Gate 4" above holds as measured: a full build of the fold has production's GPU code.

### Gate 5: preflight

`preflight.py --quality` ran on the fold image. Its interim stamp, `0.34.4-dynres-41-gfedbe05`, resolves to
`rocm7-0-34-4-dynres`. The log, verbatim:

```
profile: rocm7-0-34-4-dynres  payload=001+002+004+005+801+802+903+908
  PASS=27  SKIP=5
VERDICT: PASS
```

The run record is `runs/preflight-rocm7-0350-fold-gfedbe05.json`. `preflight_cmp.py` compares it row by row, without
timings, with production's run on its own build (#415, `runs/preflight-rocm7-0344-aspect-gb43ee8e.json`). Verbatim:

```
A: maxusai-ollama:0.34.4-rocm724-main-b43ee8e3 0.34.4-dynres-0-gb43ee8e {'PASS': 27, 'SKIP': 5}
B: maxusai-ollama:0.34.4-dynres-41-gfedbe05-rocm7-gfx1151 0.34.4-dynres-41-gfedbe05 {'PASS': 27, 'SKIP': 5}
32 rows: 30 identical, measured values included; 2 differ:
  version: actual, summary
  image_tag: actual, expected, summary
```

- The identical rows include the token and aspect ladders, `think_format`, the extraction-quality floors, the fp16
  canary (`poison_probe`), the pinned image-token budgets (3270 and 529), `payload_pin` (`161755f29`, b11081) and
  `toolchain_pin` (`rocm-7.2.4`).
- The five skips are production's too: four Metal and MLX checks, and qwen35's pinned budget.
- The `payload` line shows the patch set as the profile listed it then. This PR adds 805
  ([below](#the-rocm7-profile-widened)).

### The new surface

`fold-0350.sh` sent a System One request for two production models. It also sent one `/api/generate` request with
`"options":{"typical_p":0.9,"num_predict":8,"temperature":0}` to each image. Its log, verbatim, in selected lines
(the fold image's answer is cut at `…`):

```
systemone qwen3.6:35b-a3b-q4_k_m: {"model":"qwen3.6:35b-a3b-q4_k_m","answers":{"refund":{"type":"noul","noul":0.8867295923317952}},"usage":{"input_tokens":77,"output_tokens":1}} [HTTP 200]
systemone qwen3.8:27b-q4_K_M: {"error":"model \"qwen3.8:27b-q4_K_M\" is not supported by System One; use a local Nimble or Tev GGUF model"} [HTTP 400]
typical_p, fold image: {"model":"qwen2.5vl:3b-q4_K_M","created_at":"2026-09-30T10:31:44.976672618Z","response":"Hello! How can I assist you today","done":true,"done_reason":"length",…
time=2026-09-30T10:31:42.535Z level=WARN source=routes.go:185 msg="deprecated option provided" option=typical_p
	top_k = 40, top_p = 0.900, min_p = 0.000, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 0.900, top_n_sigma = -1.000, temp = 0.000
typical_p, promoted image: {"error":"typical_p is no longer supported"} [HTTP 400]
```

- **System One answers on qwen3.6,** production's one model on the `qwen3.5` renderer. It refuses qwen3.8 with
  upstream's message.
- **The fold applies `typical_p`, not only accepts it.** The server logs the deprecation warning, and llama-server's
  sampler takes the value (`typical_p = 0.900`). The log cuts the fold's line before its status. The server's own
  log has it: `[GIN] 2026/09/30 - 10:31:44 | 200 |  2.444729398s |      172.17.0.1 | POST     "/api/generate"`.
- The promoted image refuses the same request with a 400.

### Gate 6: think off and OCRBench

All five production models ran with thinking off. `cmp_scores.py` compares each with the baseline. From the driver's
log, verbatim:

```
0 of 980 cells differ (scores_r0344base_1_gemma4_26b-a4b-it-q4_K_M_thinkfalse.json vs scores_r0350fold_1_gemma4_26b-a4b-it-q4_K_M_thinkfalse.json)
0 of 986 cells differ (scores_r0344base_1_gemma4_31b-it-q4_K_M_thinkfalse.json vs scores_r0350fold_1_gemma4_31b-it-q4_K_M_thinkfalse.json)
0 of 982 cells differ (scores_r0344base_1_nemotron3_33b-q4_K_M_thinkfalse.json vs scores_r0350fold_1_nemotron3_33b-q4_K_M_thinkfalse.json)
0 of 978 cells differ (scores_r0344base_1_qwen3_6_35b-a3b-q4_k_m_thinkfalse.json vs scores_r0350fold_1_qwen3_6_35b-a3b-q4_k_m_thinkfalse.json)
0 of 976 cells differ (scores_r0344base_1_qwen3_8_27b-q4_K_M_thinkfalse.json vs scores_r0350fold_1_qwen3_8_27b-q4_K_M_thinkfalse.json)
```

`cmp_raw.py` compares the bytes of the stored answers (`resp_*.json`) and thinking (`think_*.txt`). Verbatim:

```
qwen3_6_35b-a3b-q4_k_m thinkfalse: answers 28 of 28 byte-identical
qwen3_8_27b-q4_K_M thinkfalse: answers 28 of 28 byte-identical
gemma4_31b-it-q4_K_M thinkfalse: answers 28 of 28 byte-identical
gemma4_26b-a4b-it-q4_K_M thinkfalse: answers 28 of 28 byte-identical
nemotron3_33b-q4_K_M thinkfalse: answers 28 of 28 byte-identical
```

- The 28 answers per model are the 27 cases and the fine-text probe.
- The first version of `cmp_raw.py` read two keys that the answer files do not have, so it counted every pair as
  identical. The figures above come from the corrected tool, which does find differences: 17 of gemma4:31b's 28
  think-on answers differ ([#427](https://github.com/MaxusAI/ollama/pull/427#issuecomment-5923052484)).

OCRBench ran on rows 0–200 with gemma4:31b, thinking off. The output is from `summarize_extbench.py --paired ocrbench
r0344base_ocr_q4 r0350fold_ocr_q4`. Its last line is a separate check, record by record, of `pred`,
`prompt_eval_count` and `eval_count`. Verbatim:

```
| model | scored | errors | empty | correct | accuracy | think | endpoint |
|---|---|---|---|---|---|---|---|
| `gemma4:31b-it-q4_K_M` | 200 | 0 | 0 | 172 | **0.86** | false | generate |
| `gemma4:31b-it-q4_K_M` | 200 | 0 | 0 | 172 | **0.86** | false | generate |

ocrbench — `echo840/OCRBench` [test], rows 0..200.

⚠ **MIXED — rows are not one campaign** (hosts: ['http://127.0.0.1:11497', 'http://127.0.0.1:11498']; builds: ['0.34.4-dynres-0-gb43ee8e', '0.34.4-dynres-41-gfedbe05'])

| pair | both ✓ | both ✗ | A only | B only | McNemar exact p |
|---|---|---|---|---|---|
| r0344base_ocr_q4 vs r0350fold_ocr_q4 | 172 | 28 | 0 | 0 | 1.000 |
OCRBench answers byte-identical: 200 of 200; prompt and answer token counts identical: 200 of 200
```

The MIXED warning names the two arms. Each arm is one build on one port.

### Gate 6: think on

The protocol is the baseline's (`baseline-c.sh`): the two-pass flow with an f16 KV cache, each model on a cold
server, over the full ladder from 16384 to 131072.
- gemma4:31b and gemma4:26b ran greedy, with the batch pinned at 2048. qwen3.6 ran greedy on its card.
- qwen3.8 and nemotron3 ran twice each at their packaged sampling, so their counts are draws.
- qwen3.6's baseline is the one that the gate's baseline uses: the 0.34.4 fold's two-pass f16 run
  (`r0344pf16_fold2p_`, 2026-09-27). That ran on `0.34.3-dynres-5-g29ae523`. Its payload differs from production's
  only by 908, and 908 changes no gfx1151 kernel.

`thinkon_read.py` reads the greedy models' finishes and `cmp_scored.py`'s quality moves. For the sampled models it
reads `rates2.py`'s pooled rates, with a two-sided Fisher exact p for each boolean rate that moved. Verbatim:

```
== gemma4_31b-it-q4_K_M (greedy): baseline r0344base_1_ against r0350fold_1_
   baseline: 27 cells, done_reason {'stop': 27}, json_valid 27, not stopped: none
   fold:     27 cells, done_reason {'stop': 27}, json_valid 27, not stopped: none
   bbox_contract: declared_ref [1920, 1080]->None
   bbox_contract_multi: declared_ref [1920, 1080]->None
   scene_single_anchored: bbox_mean_iou 0.724->0.963 B+
   quality moves: 1 favour B, 0 favour A, 2 label changes (A = scores_r0344base_1_gemma4_31b-it-q4_K_M_thinkon.json, B = scores_r0350fold_1_gemma4_31b-it-q4_K_M_thinkon.json)

== qwen3_6_35b-a3b-q4_k_m (greedy): baseline r0344pf16_fold2p_1_ against r0350fold_1_
   baseline: 27 cells, done_reason {'stop': 26, 'length': 1}, json_valid 26, not stopped: ['bbox_contract_real_1img']
   fold:     27 cells, done_reason {'stop': 26, 'length': 1}, json_valid 26, not stopped: ['bbox_contract_real_1img']
   quality moves: 0 favour B, 0 favour A, 0 label changes (A = scores_r0344pf16_fold2p_1_qwen3_6_35b-a3b-q4_k_m_thinkon.json, B = scores_r0350fold_1_qwen3_6_35b-a3b-q4_k_m_thinkon.json)

== gemma4_26b-a4b-it-q4_K_M (greedy): baseline r0344base_1_ against r0350fold_1_
   baseline: 27 cells, done_reason {'stop': 24, 'length': 3}, json_valid 24, not stopped: ['bbox_contract_anchored_1img', 'bbox_contract_real_1img', 'multi_3img_anchored']
   fold:     27 cells, done_reason {'stop': 24, 'length': 3}, json_valid 24, not stopped: ['bbox_contract_anchored_1img', 'bbox_contract_real_1img', 'multi_3img_anchored']
   quality moves: 0 favour B, 0 favour A, 0 label changes (A = scores_r0344base_1_gemma4_26b-a4b-it-q4_K_M_thinkon.json, B = scores_r0350fold_1_gemma4_26b-a4b-it-q4_K_M_thinkon.json)

== qwen3_8_27b-q4_K_M (sampled, two runs per arm): rates2.py base=r0344base_r1/r2 fold=r0350fold_r1/r2
                                            base             fold   (blocks 54 vs 54)
     anchor_beats_declared            0/40             0/40      
     anchor_present                  22/40            22/40      
     contract_followed               40/40            40/40      
     declaration_matches_boxes       40/40            40/40      
     declaration_valid               40/40            40/40      
     invoice_no                       2/2              2/2       
     json_valid                      54/54            54/54      
     q1_right                         4/4              4/4       
     q2_right                         4/4              4/4       
     q4_bbox_hit                      2/4              2/4       
     self_check                      22/22            22/22      
     serial_found                     6/6              6/6       
     think                           54/54            54/54      
     total_right                      2/2              2/2       
     answer_chars                        852.370          847.704
     bbox_mean_iou                         0.979            0.984
     eval_count                         1234.426         1331.111
     hits_anchor                           3.300            3.250
     hits_bestfit                          5.125            5.225
     hits_declared                         6.000            5.950
     iou_anchor                            0.531            0.526
     iou_at_implied_scale                  0.951            0.947
     iou_declared                          0.963            0.952
     labels_found                          6.000            5.957
     name_bbox_mean_iou                    0.770            0.787
     thinking_chars                     1859.630         2069.815
     done_reason                    {'stop': 54}     {'stop': 54}
     NOT CONVERGED                             -                -

== nemotron3_33b-q4_K_M (sampled, two runs per arm): rates2.py base=r0344base_r1/r2 fold=r0350fold_r1/r2
                                            base             fold   (blocks 54 vs 54)
     anchor_beats_declared            9/40             6/40      
     anchor_present                  21/40            20/40      
     contract_followed               21/40            18/40      
     declaration_matches_boxes       21/40            18/40      
     declaration_valid               25/40            22/40      
     invoice_no                       2/2              2/2       
     json_valid                      53/54            54/54      
     q1_right                         4/4              4/4       
     q2_right                         4/4              4/4       
     q4_bbox_hit                      4/4              4/4       
     self_check                      10/21            11/20      
     serial_found                     3/6              3/6       
     think                           54/54            54/54      
     total_right                      2/2              2/2       
     answer_chars                       1236.056         1233.907
     bbox_mean_iou                         0.495            0.659
     eval_count                         8457.667         5144.167
     hits_anchor                           2.700            2.600
     hits_bestfit                          5.225            4.975
     hits_declared                         3.325            3.000
     iou_anchor                            0.288            0.261
     iou_declared                          0.384            0.317
     labels_found                          5.935            5.870
     name_bbox_mean_iou                    0.092            0.004
     thinking_chars                    19684.222        12061.685
     done_reason                {'stop': 53, 'length': 1}     {'stop': 54}
     NOT CONVERGED                  ['finetext']                -
   Fisher p, anchor_beats_declared: 0.568
   Fisher p, anchor_present: 1.000
   Fisher p, contract_followed: 0.655
   Fisher p, declaration_matches_boxes: 0.655
   Fisher p, declaration_valid: 0.650
   Fisher p, json_valid: 1.000
   Fisher p, self_check: 0.758
```

`cmp_raw.py` on the greedy models, the first line of each, verbatim:

```
gemma4_31b-it-q4_K_M thinkon: answers 11 of 28 byte-identical; thinking 11 of 28 byte-identical
qwen3_6_35b-a3b-q4_k_m thinkon: answers 28 of 28 byte-identical; thinking 28 of 28 byte-identical
gemma4_26b-a4b-it-q4_K_M thinkon: answers 28 of 28 byte-identical; thinking 28 of 28 byte-identical
```

- **No model loses a finish.**
  - gemma4:31b and qwen3.8 finish every case in both arms.
  - nemotron3's fold finishes all 54 blocks. Its baseline lost one sampled block to a loop (below).
  - qwen3.6 leaves `bbox_contract_real_1img` unfinished in both arms. That is the known pixel-coordinate loop.
  - gemma4:26b leaves the same three cases unfinished in both arms: `multi_3img_anchored`,
    `bbox_contract_anchored_1img` and `bbox_contract_real_1img`. Each runs to the cap, 122,880 tokens at 131072.
- **qwen3.6 and gemma4:26b are byte-identical:** all 28 answers and all 28 thinking texts on each model, the loops
  included.
- **gemma4:31b** has one quality move, and it favours the fold: `scene_single_anchored`'s IoU goes from 0.724 to
  0.963. This case is unstable. It scored 0.965 under `q8_0`, and the f16 baseline dropped it to 0.724. Both arms
  here run the same GPU code with an f16 cache, so this move comes from the run's state (below).
  The two label changes are in `bbox_contract` and `bbox_contract_multi`. There the fold's answer gives no reference
  size, where the baseline's gave 1920×1080.
- **qwen3.8:** every boolean rate is identical, and all 108 blocks end with `stop`.
- **nemotron3:** the boolean rates move in both directions, by 1 to 3 blocks each, all at Fisher p ≥ 0.57. The
  largest move against the fold is `contract_followed`, 21/40 in the baseline and 18/40 in the fold (p = 0.655).
  - The fold finishes all 54 blocks. The baseline's one loop, `finetext` in its second run, does not recur. One
    draw in 54 cannot show a fix.
  - The mean `eval_count` falls from 8,458 to 5,144 tokens, mostly because of that loop: 122,880 tokens in one
    block. Without `finetext`, the means are 6,024 and 5,228.
  - Per case, the baseline's mean `eval_count` is the higher one in 12 of 26 cases, and the fold's in 14. A sign
    test gives p = 0.85, and a permutation test of the per-case means gives p = 0.12. The baseline's own two runs
    differ per case by 2,058 tokens on average, and the two arms by 2,180. So the gap comes from the draws.

### gemma4:31b in the suite: the two-pass flow's slot race

gemma4:31b runs greedy. Both arms send the same prompts (the same `prompt_sha` and `prompt_eval_count`), the same
sampler and the same llama-server flags (`-b/-ub 2048`, `-np 2`, `-c 32768`, f16, flash attention on). Yet its
thinking differs in 17 of 28 cells. In run order the first cell is identical, and the differences start at the
second.

Step 2 of the `kv-loop-check` skill separates the case from the run's history. `cold0350.sh` captured three of the
differing cells cold with `thinkcap.py`, on each image in turn. It used the suite's first rung: 16384, the two-pass
flow, f16, flash attention on, two slots and the batch pinned at 2048, on `:11494`. `cold/compare.txt`, verbatim:

```
bboxm_free_noanc_named:
  cold prod vs cold fold: thinking BYTE-IDENTICAL (942 chars), answer identical; eval 850|850, prompt_eval 1396|1396, done stop|stop
  cold prod vs suite base (r0344base_): differs at 208 (882 chars)
  cold prod vs suite fold (r0350fold_): differs at 0 (660 chars)
bboxm_pin_anc_pos:
  cold prod vs cold fold: thinking BYTE-IDENTICAL (2050 chars), answer identical; eval 1505|1505, prompt_eval 1516|1516, done stop|stop
  cold prod vs suite base (r0344base_): differs at 1770 (2193 chars)
  cold prod vs suite fold (r0350fold_): differs at 58 (2392 chars)
scene_single_anchored:
  cold prod vs cold fold: thinking BYTE-IDENTICAL (2358 chars), answer differs; eval 1628|1626, prompt_eval 1825|1825, done stop|stop
  cold prod vs suite base (r0344base_): differs at 960 (7944 chars)
  cold prod vs suite fold (r0350fold_): differs at 586 (2978 chars)
```

The server logs show which slot each pass took. `cold/slots.txt`, verbatim:

```
prod:
  capture 1 pass 1: slot 1 by LRU
  capture 1 pass 2: slot 0 by LRU
  capture 2 pass 1: slot 1 by LRU
  capture 2 pass 2: slot 0 by LRU
  capture 3 pass 1: slot 1 by LRU
  capture 3 pass 2: slot 1 by LCP similarity
fold:
  capture 1 pass 1: slot 1 by LRU
  capture 1 pass 2: slot 0 by LRU
  capture 2 pass 1: slot 1 by LRU
  capture 2 pass 2: slot 0 by LRU
  capture 3 pass 1: slot 1 by LRU
  capture 3 pass 2: slot 0 by LRU
```

- **Cold, the fold's pass-1 thinking equals production's, byte for byte, in all three cells.** That includes
  `bboxm_free_noanc_named`, whose thinking in the suite differed from its first character.
- **Both suites leave the cold trajectory.** The baseline's suite leaves it at characters 208, 1770 and 960, and the
  fold's at 0, 58 and 586.
- **Pass 2 goes to either slot, by timing.** llama-server gives pass 2 the idle slot whose cache matches its prompt
  best ("by LCP similarity"). That is pass 1's slot, if pass 1 has released it. If not, pass 2 takes the least
  recently used idle slot ("by LRU"), which is the other, empty one. Production's own image did both in three cold
  captures, so the race is not the fold's.
- Where pass 2 took the same slot on both images (captures 1 and 2), the answers are identical. Where it did not
  (capture 3), they differ by 2 tokens.
- The suite shows pass 2 diverging by itself too: `bbox_contract`'s thinking is byte-identical in both arms, and its
  answer differs.
- **So in the suite, gemma4:31b's two-pass think-on does not repeat byte for byte on gfx1151, on either image.** An
  inference: each cell's slot choice changes what the next cell finds in the slots and the prompt cache. So think-on
  is compared here by finishes, rates and quality moves, not by bytes.
- The 0.34.4 record inferred this mechanism from slot logs
  ([upstream-sync-0.34.4.md](upstream-sync-0.34.4.md#gate-6-think-on-under-the-aligned-protocol)). Production's own
  image now shows both outcomes.
- Here the race shows on gemma4:31b only. qwen3.6's and gemma4:26b's suites repeat byte for byte (above).

### The rocm7 profile, widened

`expectations.toml`'s `rocm7-0-34-4-dynres` now admits this fold's builds:
- `version_pattern` is `^0\.3(4\.[34]|5\.0)-dynres-\d+-g[0-9a-f]{7,40}$`. ADR 0032 widens a profile for a fold that
  moves no native input, and this fold moves none. 0.35.1 stays refused, because its rc0 moves llama.cpp to b11232.
- `patchset` lists 805. Every build since #371 carries it, as the builds before it came to carry 908.
- `TestRocm7ProfileAdmitsTheV0350Fold` pins both changes. It admits production's stamp and the fold's interim and tag
  stamps. It refuses 0.35.1, an earlier payload, a dirty tree and a point tag. `test_verdicts.py` passes 200 tests.

## Gates 4–6 on Apple Silicon (2026-10-01)

**Host.** The Metal host, `macbook-pro-m5-max-128GB/mlx-metal`: an Apple M5 Max with 40 GPU cores (Metal 4) and
128 GiB of unified memory, on macOS 26.6.2.
- **Its profile** ([ADR 0046](../adr/0046-a-published-result-carries-a-profile-of-its-machine.md)), captured during
  the preflight run, is
  [`preflight-mlx-metal-0350-release-g043f441.host-profile.json`](../vision-suite/preflight/runs/preflight-mlx-metal-0350-release-g043f441.host-profile.json).
  The capture taken during gate 6 differs from it only in `collected_at` and in the models filesystem that `--fs`
  names.
- The maintainer started this leg on 2026-10-01. It ran from 21:40 AEST to 23:34.
- Every gate ran on a scratch server in production's environment: the launchd plist's three variables
  (`OLLAMA_FORMAT_TWO_PASS=1`, `OLLAMA_KV_CACHE_TYPE=f16`, `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`) and its MLX store,
  with one model loaded at a time. Preflight and the smoke checks used `127.0.0.1:11437`, and gate 6 used `:11436`.
- Production on `:11435` kept serving 0.34.4. Every GPU step waited until production had been idle 15 minutes, in
  High Power mode, and would have stopped and re-run had it woken. It never woke: the contention ledgers
  have no lines, and its log shows no model request from 21:38 to 23:35.
- **The reference is production's own build,** `0.34.4-dynres-0-gb43ee8e`: its preflight run of 2026-09-28, and its
  think-off and OCRBench measurements of 2026-09-29, `r0344rel_`
  ([upstream-sync-0.34.4.md](upstream-sync-0.34.4.md#deployed-and-verified-on-the-build-itself-metal-2026-09-29)).
  Gate 6 ran the same suite tree, driver logic, order and settings, on the same model files: every manifest it reads
  predates that baseline.
- The scripts and tools named below are in the Metal host's run directory. `leg-0350.sh` ran the gates in order.

**On Apple Silicon the fold changes no result that these gates measure.**
- Its payload is production's, byte for byte. The binary is the only new file.
- Its preflight repeats the deployed build's run in 32 of the 35 rows both runs have, every measured value included.
  The version row names the build, and two drafted `think_format` rows vary as production's own two runs of 0.34.4 do.
- With thinking off, all 196 answers on seven models equal the deployed build's, byte for byte, and so do their
  token counts. So do all 200 OCRBench answers.
- The MLX tests pass as on the 0.34.4 deploy. System One and per-request `typical_p` work as upstream describes, and
  `typical_p` reaches llama-server's sampler.

### Gate 4: a Go-only build on production's payload

- **The build:** `go build` at `v0.35.0-dynres` (`043f441a7`), with `build-macos.sh`'s flags and stamp. It stamps
  `0.35.0-dynres-0-g043f441`, and its build info reads `vcs.modified=false`.
- **The payload is production's.** The stage's `lib/ollama` is a copy of the archived 0.34.4 pairing, which is
  byte-identical to the `build/lib/ollama` that production serves: 19 files, 231 MB.
- So "Gate 4" above holds on Metal as measured. Since `b43ee8e37`, the only native change is 805, a switch in the
  CUDA flash-attention code, which the Metal payload does not build. MLX (`59d600b5`), MLX-C and llama.cpp (`b11081`)
  are unchanged.
- In the server log, the engine reports `"MLX version"=0.32.2-65-g59d600b`, and XGrammar v0.2.7 loads from the
  stage's own `lib/ollama/mlx_metal_v4/`.

### Gate 5: preflight

`preflight.py --quality` ran on the release build. Its tag stamp resolves only through this record's widening of
`mlx-metal-0-34-4` ([below](#the-mlx-metal-profile-widened)). The log, verbatim:

```
profile: mlx-metal-0-34-4  payload=
  PASS=23  SKIP=16
VERDICT: PASS
```

The run record is `runs/preflight-mlx-metal-0350-release-g043f441.json`. `pf_cmp.py` compares it row by row, without
timings, with the release run of production's build (2026-09-28). Verbatim:

```
A: 0.34.4-dynres-0-gb43ee8e profile mlx-metal-0-34-4 mlx 0.32.2-65-g5 llama.cpp 161755f29 {'PASS': 23, 'SKIP': 12}
B: 0.35.0-dynres-0-g043f441 profile mlx-metal-0-34-4 mlx 0.32.2-65-g5 llama.cpp 161755f29 {'PASS': 23, 'SKIP': 16}
39 rows: 32 identical, measured values included; 7 differ:
  version : expected: "^0\\.34\\.[34]-dynres-\\d+-g[0-9a-f]{7,40}$" -> "^0\\.3(4\\.[34]|5\\.0)-dynres-\\d+-g[0-9a-f]{7,40}$"; actual: "0.34.4-dynres-0-gb43ee8e" -> "0.35.0-dynres-0-g043f441"
  think_format gemma4_unified: actual: {"response_chars": 225, "thinking_chars": 1658, "eval_count": 472} -> {"response_chars": 273, "thinking_chars": 803, "eval_count": 243}
  think_format qwen35: actual: {"response_chars": 310, "thinking_chars": 779, "eval_count": 269} -> {"response_chars": 310, "thinking_chars": 779, "eval_count": 267}
  extraction_quality gemma4: only in B
  extraction_quality gemma4_unified: only in B
  extraction_quality qwen35moe: only in B
  extraction_quality qwen35: only in B
```

- **The version row** names the build and the widened pattern.
- **The two `think_format` rows are drafted requests.** Under the two-pass flow, pass one drafts, and how deep it
  drafts follows wall-clock time. Production's own two runs of 0.34.4, release and post-deploy, differ in the same two
  rows.
- **The four `extraction_quality` rows come from `--quality`,** which the 2026-09-28 runs did not pass. They skip,
  because the profile records no quality thresholds.
- The identical rows include every token and aspect ladder, `payload_pin` (`161755f29`, b11081), `mlx_payload_pin`
  (`0.32.2-65-g59d600b`), the M5 tensor checks, and `think_format` on gemma4 and qwen35moe.

### The native gate

`go test ./mlxrunner/... -p 1` with `OLLAMA_VISION_E2E=1` ran in this tree on production's payload: 806 passed,
0 failed and 1 skipped, as on the 0.34.4 deploy. `TestVisionEndToEnd` and `TestVisionGoldenParity` pass on real
weights. The skip is `TestGlobalScaleSurvivesStorageBitExactly`: its fixture's float32 round trip happens to be exact,
so it cannot witness the rule.

### The new surface

`step-smoke.sh` sent one think+format request to gemma4:26b-nvfp4, two System One requests and one `/api/generate`
request with `"options":{"typical_p":0.9,"num_predict":8,"temperature":0}`. Its log, verbatim, in selected lines (the
`typical_p` answer is cut at `…`):

```
content parses as JSON
speculative decode stats lines: before=0 after=1
time=2026-10-01T21:43:14.550+10:00 level=INFO source=speculate_stats.go:62 msg="speculative decode stats" iterations=64 drafted=126 accepted=107 acceptance=0.85 avg_draft=1.97 max_draft=4 avg_accepted=1.67 depth_over_time="0.5/1 1.2/2 2.0/2 2.1/3 2.4/3 2.4/3 3.2/4 1.9/3"
{"model":"qwen3.6:35b-a3b-q4_K_M","answers":{"refund":{"type":"noul","noul":0.9514042656271932}},"usage":{"input_tokens":66,"output_tokens":1}} [HTTP 200]
{"error":"model \"qwen3.8:27b-q4_K_M\" is not supported by System One; use a local Nimble or Tev GGUF model"} [HTTP 400]
{"model":"qwen3.6:35b-a3b-q4_K_M","created_at":"2026-10-01T11:43:20.322638Z","response":"Hello! How can I help you today","done":true,"done_reason":"length",… [HTTP 200]
time=2026-10-01T21:43:20.176+10:00 level=WARN source=routes.go:185 msg="deprecated option provided" option=typical_p
	top_k = 20, top_p = 0.950, min_p = 0.000, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 0.900, top_n_sigma = -1.000, temp = 0.000
```

- **The two-pass flow works as deployed.** Pass one drafted, pass two under the grammar did not, and the answer
  parses as JSON.
- **System One answers on qwen3.6,** production's one model on the `qwen3.5` renderer. It refuses qwen3.8 with
  upstream's message, as on gfx1151.
- **The fold applies `typical_p`.** The server logs the deprecation warning, and llama-server's sampler takes the
  value (`typical_p = 0.900`).

### Gate 6: think off and OCRBench

All seven think-off cells ran on the release build as the deployed build's did on 2026-09-29: a cold start per cell,
`num_ctx` 16384 and `num_predict` 2200. Think-off requests carry a grammar and do not draft, so their answers repeat
byte for byte. `eq_check_0350.py` compares the stored answers (`resp_*.json`: the 27 cases and the fine-text probe) and
their `eval_count`. Verbatim:

```
think off, per model: A = r0344rel_1, B = r0350fold_1
model                         answers identical  eval_count equal  differing cases
gemma4_12b-nvfp4                       28/28             28/28     -
gemma4_26b-nvfp4                       28/28             28/28     -
gemma4_31b-nvfp4                       28/28             28/28     -
qwen3_6_35b-a3b-nvfp4                  28/28             28/28     -
qwen3_8_27b-nvfp4                      28/28             28/28     -
gemma4_31b-it-q4_K_M                   28/28             28/28     -
qwen3_6_35b-a3b-q4_K_M                 28/28             28/28     -
all cells: 196/196 answers byte-identical
```

- The five MLX models and the two GGUF ones are all byte-identical.
- Both builds launched llama-server for gemma4:31b-it-q4_K_M with `-b 2048 -ub 2048`. So the automatic batch, which
  follows free memory and can move gemma4's GGUF answers, did not move here.

OCRBench ran on rows 0–199 with gemma4:31b-nvfp4 (manifest `637cc0ff1570`): generate, thinking off, `num_ctx` 16384,
temperature 0. OCRBench carries no grammar, so gemma4 drafts. `ocr_cmp.py` pairs the records with the deployed build's
chunk of 2026-09-29. Verbatim:

```
A: ocrk0344rel_c0 gemma4:31b-nvfp4 server ['0.34.4-dynres-0-gb43ee8e'] host ['http://127.0.0.1:11436'] rows 0..199: 174/200 correct, errors 0, empty 0
B: ocrk0350fold_c0 gemma4:31b-nvfp4 server ['0.35.0-dynres-0-g043f441'] host ['http://127.0.0.1:11436'] rows 0..199: 174/200 correct, errors 0, empty 0
paired rows: 200 (rows in one arm only: 0)
both correct 174, both wrong 26, A only 0, B only 0, McNemar exact p = 1.000
questions and golds equal: 200/200
predictions byte-identical: 200/200; prompt and answer token counts identical: 200/200
no record differs
```

- The drafted short answers repeat, as they did between the 0.34.4 fold and its deployed build: 200 of 200.

### Think on: not run

Think on did not run on this host. On Metal, gemma4 drafts while it thinks, and a long drafted answer does not repeat
byte for byte even on one build ([#375](https://github.com/MaxusAI/ollama/pull/375#issuecomment-5824799988)). So
think on could be compared only by finishes and rates, as on gfx1151, where that comparison found sampling draws and
the two-pass flow's slot race and nothing else. The fold puts nothing new on that path. Its server changes are the
System One route and `typical_p`'s acceptance, and the fork's own code, the two-pass flow included, carries over line
for line ([The conflict and its resolution](#the-conflict-and-its-resolution)). The maintainer can still ask for the
run.

### The mlx-metal profile, widened

`expectations.toml`'s `mlx-metal-0-34-4` now admits this fold's builds:
- `version_pattern` is `^0\.3(4\.[34]|5\.0)-dynres-\d+-g[0-9a-f]{7,40}$`, the shape `rocm7-0-34-4-dynres` took. ADR
  0032 widens a profile for a fold that moves no native input, and this fold moves none. 0.35.1 stays refused, because
  its rc0 moves MLX to `64ea011c` and llama.cpp to b11232.
- `patchset` stays empty. llama.cpp's compat patches, 805 included, do not apply to MLX.
- `TestMlxMetalProfileAdmitsTheV0350Fold` pins the widening. It admits production's stamp and the fold's interim and
  tag stamps, and refuses 0.35.1, an earlier payload, a dirty tree, a point tag and the retired native stamp. Through
  the resolver, the tag stamp maps to this profile. The old pattern fails the admit test. `test_verdicts.py` passes
  211 tests.
