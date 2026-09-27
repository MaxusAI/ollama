# TASK: fold upstream v0.34.4 into main

Upstream [v0.34.4](https://github.com/ollama/ollama/releases/tag/v0.34.4) — fourteen commits, 70 files,
+2,074/−912 — folded on top of `main` at `ba7150428` (#373, which carries the v0.34.3 fold). Branch
`task/upstream-sync-0.34.4`, worktree `claude-scratch/wt-sync0344` on the CUDA host (`ai-server/mlx-cuda`).

**In progress on the CUDA host — please do not start a parallel 0.34.4 fold.** ROCm and Metal: once the merge
lands on this branch, your gate 4 and gate 6 legs can build from it.

## Status (2026-09-25)

| gate | state |
|---|---|
| 1, the merge | **done** — `c3e393d56`; 15 conflicted files. `go build` and `go vet` clean over all 80 packages; `go test` 58 packages ok, 0 failed; `server` green with `OLLAMA_FORMAT_TWO_PASS` unset **and** set. MLX tests skip here until gate 4 builds the payload |
| 2, docs and paths | **done** — `check_source_paths.py` clean over the 66 files the fold wrote |
| 3, the patch series | **done** — all eight (001 002 004 005 801 802 903, and 908 from 2026-09-26) apply clean to `b11081` on a real checkout, in order, and 908 reverse-applies, so a re-configure is safe; served projectors unchanged |
| 4, image | **done on CUDA** — `e8f7a2a1968c`, a full build. MLX tests on its payload 889 passed, 0 failed; the vision goldens identical to 0.34.2's; **done on gfx1151**: `0.34.3-dynres-5-g29ae523-rocm7-gfx1151`, with a b11081 payload whose structure is unchanged against 0.34.3's; rebuilt with 908 as `0.34.3-dynres-22-g5584539`, and **908 changes no gfx1151 kernel** |
| 5, preflight | **PASS on CUDA**: run 2 PASS=21 SKIP=8, with the pins moved in `c79e50d98` after run 1 (FAIL=2 on the two pins, by design); run 3, on the image with 908, the same. **gfx1151: PASS=20 SKIP=12** with the new `rocm7-0-34-4-dynres` (#378), and the same on the 908 image |
| 6, campaigns | **done on CUDA** (2026-09-27). GGUF think-off: 210 of 6,909 cells move, in the head-dimension-256 models only, and reverting `ce8caa6e6` restores production on both probes: its device half on gemma4:31b, its host half on qwen3.6. On the image with 908, gemma4:31b, 26b and e4b and nemotron3 equal production; e2b and the qwen models keep the host half's movement. MLX think-off: no consistent difference, inside or across MLX-CUDA's run-to-run spread. OCRBench: production's scores, but for one reproducible item on GGUF q4, which is `ce8caa6e6`'s device half. Think-on: `ce8caa6e6`'s device half leaves 6 of 27 gemma4:26b cases in loops that never end, against 1 without it; 31b is unaffected. MLX think-on: the single pass loops no more than two-pass, with or without a fixed history. gemma4:26b `multi_3img_anchored` never converges in any of 8 runs. Item 8: its trap sentence alone loops it on GGUF on the fold image, not on the image that ships; on MLX it finishes about one cold draw in five, with the sentence or without it. Drafting: under production's knob the single pass never drafts, so it thinks 1.5–1.7× slower, and the deploy runs two-pass (open item 7). See [Gates 4–6 on CUDA](#gates-46-on-cuda-2026-09-25). **gfx1151:** think-off and OCRBench equal production in every scored cell. Think-on under the aligned protocol is done (2026-09-27): on all five GGUF models the single pass leaves the same cases unfinished as two-pass, and those loops come from the prompt (#387). See [Gates 4–6 on gfx1151](#gates-46-on-gfx1151-2026-09-25) |
| tag and deploy | **not yet.** The CUDA deploy is prepared. It mirrors production (`0.34.2-dynres-0-g5bffaac`) and adds `OLLAMA_KV_CACHE_TYPE=f16` and `OLLAMA_FORMAT_TWO_PASS=1` (open items 6 and 7). It waits on the merge, the `v0.34.4-dynres` tag, the release image and the maintainer's word |

**Three hosts converged on this merge.** The ROCm and Metal hosts had each started the same fold before #375
existed, stopped, and cross-checked instead; see [#375](https://github.com/MaxusAI/ollama/pull/375).

## What v0.34.4 changes for the fork

- **llama.cpp `b10969` → `b11081`.** 112 builds. Gate 3 applies the whole series (001, 002, 004, 005, 801, 802,
  903) with plain `git apply` on a real checkout first, and diffs the projector cases of every served model.
- **MLX `d9add9d1` → `59d600b5`**, the first MLX move since `d9add9d1`. The vision goldens are re-taken against
  the archived artifacts, and the kernel fixes in the range are checked against the served models' dimensions.
- **XGrammar 0.2.5 → 0.2.7.** 0.2.5's `allOf` with more than one branch enforces nothing; re-checked on 0.2.7.
- **Structured outputs after thinking** (four commits across `server/`, `llm/`, `mlxrunner/`, `model/parsers/`)
  — meets the fork's ADR 0004 marker flow and ADR 0033 grammar passthrough.
- **`mlx: select Gemma 4 image resolution dynamically`** (#18603) — a policy change, see below.
- **`mlx: speed up Qwen 3.8 prompt processing`** (#18550) — a fused, scale-deferring SwiGLU, see below.

## Resolved so far

### gemma4 on MLX: the fork's pipeline, unchanged

Upstream replaces the checkpoint's fixed budget with a per-image choice among 70/140/280/560/1120 "closest to
the input resolution … without adding an API parameter". The fork's contract is the opposite shape: per-request
`image_min_tokens`/`image_max_tokens`, defaults 70/1120, **fill** the budget and snap to the ladder, shared with
the GGUF path through `llm.BudgetFillSize` (ADR 0003, 0008, 0021). ADR 0008 exists because under-filled gemma4
grids measurably lost grounding. So the seven gemma4 files resolve to the fork's pipeline — byte-identical to
`main` — and upstream's `process_image.go` stays deleted.

Upstream's commit also adds a guard that rejects a grid overrunning the learned position table. On every served
gemma4 checkpoint the table is 10,240 per axis and the fork's 1,120-token ceiling bounds the longest side at
1,120 × 3 = 3,360 patches, so the guard is unreachable here. **Adopting upstream's per-image budget as a default
is a policy decision for an ADR with a measurement, not a merge resolution**; it is not made in this fold.

### `mlx/ops_extra.go`: upstream's scale helpers, in ADR 0039's terms

Upstream still stores global scales in MLX's `m × 2688` form, and #18550 adds `globalScaleFactor(s) = s / 2688`
and `identityGlobalScale() = 2688`, used by `scaleAndCast` and by a new fused `SwiGLUScaled` that defers each
projection's scale out of the matmul. This fork stores the checkpoint's own `m` (ADR 0039). **The conflicted line
was the easy part: `mlx/act.go`'s call to `globalScaleFactor` auto-merged cleanly**, and under `m`-valued scales
it would scale every deferred nvfp4 gate and up projection by 1/2688 — and compile.

Resolved by defining the two helpers in ADR 0039's terms — the stored `m` already is the multiplier, the identity
is 1 — so every upstream call site is correct as written. Upstream's two new tests could not see this: each
compares two paths that share `globalScaleFactor`, so they agree under either representation. Their fixtures now
use the stored form, and `TestSwiGLUScaledMatchesSeparateScaling` gains an assertion that applies the factor
directly, which is the check that fails if the division comes back.

### Think+format: single pass by default, two-pass behind a switch

Upstream replaced the two-pass structured-output flow with a single pass on both engines: the parsers report the
strings that end their thinking (`ThinkingClose`), the server names them on one completion request, and each runner
constrains only what follows — llama-server through a GBNF wrapper around its own schema conversion, MLX through an
XGrammar structural tag. It removes the second prefill, the dropped boundary chunk, and MLX's stray first token in the
JSON, and it covers all 18 parsers where the fork's marker hook covered two.

**The maintainer's decision: single pass is the default, and `OLLAMA_FORMAT_TWO_PASS=1` keeps ADR 0004's flow as the rollback**
if single pass regresses on a served model.

- **`routes.go` is resolved by function, not by hunk.** Both handlers were rewritten too deeply: taking the fork's
  side of each hunk left upstream's deletions *between* the hunks, including the `structuredOutputsState` type the
  kept code uses. So the file is `main`'s handlers plus upstream's two changes outside them — `getExistingName`
  (#18438) and the `thinkingCloseForCompletion` helper — and the switch gates the fork's own defer decision
  (`deferViaMarker`/`deferViaTransition` in Generate, `deferring` in Chat). Closing strings go on the request only
  when it is off; with it on, pass two's prompt already ends past the marker.
- **The two-pass hooks come back in the lower layers.** Upstream deleted `IncludeIntermediateMetrics` from
  `llm.CompletionRequest`, llama-server's `TimingsPerToken` and per-chunk metrics, and the MLX request literal —
  every one in a file that merged without a conflict. All are restored and inert unless the switch is on.
- **llama-server:** upstream's block landed inside the fork's `runCompletionPhase`, which returns `(result, err)`,
  and returned a bare `err`. The ROCm host flagged this before it bit.
- **MLX:** upstream's thinking-aware structural tag wraps the fork's whitespace-bounded `json_schema` element, so the
  bound holds on both branches; the plain tag is byte-identical to `main`'s.
- **Tests pin the mode they test.** The nine two-pass tests set `OLLAMA_FORMAT_TWO_PASS=1`; they were found by
  running each candidate alone under the default (eight fail, one hangs waiting for a second request). The
  single-pass route tests, taken from the ROCm host's cross-check, pin it unset. With the switch on they fail with
  "got 2 completion calls, want 1", which is the switch visibly changing the flow.

**What single pass changes, for the gates** (from the ROCm host's review):

1. `num_predict` now bounds the total output; the two-pass total could exceed it (8290 vs 8192 on qwen3.6
   `bbox_contract_reasoning`).
2. With format, tools and thinking together, a tool call can no longer replace the formatted answer (harmony excepted).
3. Raw generate never defers: the format applies from token 0.
4. An EOS inside the thinking returns only the thinking, `response:""`, `done_reason:"stop"`.
5. **On MLX a think+format request carries a grammar from its first token**, so with production's
   `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0` it never drafts during the thinking, where pass one used to. Measure at 0 and 1.
   Measured (the drafting probe, under gate 6): the single pass then thinks 1.5–1.7× slower than two-pass does. The
   CUDA deploy therefore runs two-pass (open item 7).

Think-off cells cannot tell the two flows apart — the suite always sends `format:"json"` and both constrain from
token 0 when thinking is off — so gate 6 needs the think-on `bbox_contract_reasoning` cells. nemotron3 has no
sampling card, so its think-on cells run at packaged defaults and compare as rates, not cells.

## Cross-checks from the other hosts

| host | tree | result |
|---|---|---|
| ROCm, gfx1151 | its own merge, `dd19f1202` | gemma4 and `ops_extra.go` resolved identically; `requestGrammar` byte-identical. b11081 and all seven patches build on HIP (rocm7 and rocm10). Single pass end to end on llama-server: gemma4:e2b and qwen3:0.6b 5/5; the grammar is transparent to the thinking in matched cache state |
| Metal, M5 Max | the ROCm tree | **MLX tests where they execute: 900 passed, 0 failed, 4 stated skips.** XGrammar 0.2.7's standalone CMake builds clean on macOS. The ADR 0039 helper fix reached a third time; against upstream's helpers the contract test reads `SwiGLUScaled()[0] = -5.6e-07, want -0.0122` while upstream's own test passes all five cases |

One correction between the readings: the ROCm review lists `tools/mtmd` as unchanged in the range. `git diff
b10969 b11081 -- tools/mtmd/` shows one hunk, `clip.cpp` checking that the compute graph allocated. Its conclusion
stands, since that is error handling rather than preprocessing.

## Gates 4–6 on CUDA (2026-09-25)

GPU0 of the CUDA host (RTX PRO 6000 Blackwell, sm_120), shared throughout with another long-running workload, so every
arm waited for its model's measured footprint plus the host's 16 GiB reserve to stay free for three minutes before it
started. Production on `:11497` was not touched.

### Gate 4: the image

A full build: llama.cpp and MLX both move, so a Go-only swap was not valid. 2 h 33 min through the big-disk builder.

| | candidate | production |
|---|---|---|
| image | `maxusai/ollama:sync-0.34.4`, `e8f7a2a1968c`, 5.43 GB | `maxusai/ollama:sync-0.34.2-main` |
| stamp | `0.34.3-dynres-5-g29ae523` | `0.34.2-dynres-0-g5bffaac` |
| `llama-server --version` | commit `161755f29` (b11081) | commit `391fac164` (b10969) |
| payload | 2,696 files: `cuda_v12 cuda_v13 include llama-quantize llama-server mlx_cuda_v13 vulkan` | the same count and directories |
| `mlx_cuda_v13/libmlx.so` | `2c2ddb72e09344f9` | `b8d8427e3300be03` |
| `cuda_v13/libggml-cuda.so` | `4d1348cdf1fbd067` | `430dfca96e9e2864` |

Hashes are the first 16 hex digits of the SHA-256.

- **MLX tests on the built payload: 889 passed, 0 failed, 12 skipped**, every skip for a stated reason.
- **The vision goldens are identical to the 0.34.2 fold's in every printed digit**: gemma4:12b-nvfp4 mean −0.02562,
  std 2.4605, norm_mean 151.309, max sampled delta 0.0625; gemma4:31b-nvfp4-tower4bit −0.00621, 1.3243, 96.999,
  0.0898. The MLX move leaves the vision towers' numerics where they were.
- `TestMulGatherQMMGlobalScale` is gated to Metal upstream. With the gate widened it passes on CUDA (128.8 s); the
  widening is its own change (open items).

### Gate 5: preflight

Run 1, profile `cuda-dynres-903`, candidate in a canary container: **FAIL=2 PASS=19 SKIP=8, and the two failures are
the pins**, which fail by design when a payload moves: `payload_pin` (llama.cpp) and `mlx_payload_pin`. Everything the
profile measures passes on all three arches (nemotron_h_omni, gemma4, qwen35): the token ladders 5/5 within ±2, the
pinned budgets (3328 → 3270 and 560 → 529), the text baselines (19, 19, 13), the poison probe, and `think_format`,
which now runs the single pass: valid JSON after thinking at 486, 161 and 324 tokens.

The pins moved in `c79e50d98` with run 1 as provenance: `llama_cpp_build = "161755f29"`, `mlx_build =
"59d600b5e64c238427d0f8d897ab7c682ef4d3d2"`. **Run 2: VERDICT PASS, PASS=21 SKIP=8.** Both pins pass, and every measured
check reproduces run 1, down to `think_format`'s 486, 161 and 324 tokens.

### Gate 6: what was compared

| arm | image | stamp |
|---|---|---|
| control | production's image, `maxusai/ollama:sync-0.34.2-main` | `0.34.2-dynres-0-g5bffaac` |
| candidate | `maxusai/ollama:sync-0.34.4`, `e8f7a2a1968c` | `0.34.3-dynres-5-g29ae523` |

Every arm ran in a canary container with production's environment (`OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`) plus
`OLLAMA_MAX_LOADED_MODELS=1` and a 16 GiB `OLLAMA_GPU_OVERHEAD`, a cold container per model, and the CUDA and MLX JIT
caches persisted per architecture. The suite is `run_engine_compare.sh`, think off, `format` on, ladder 8192 → 65536.
Cells are every scalar score field of every block, provenance and timing excluded (`cellcmp-0344.py`); "better" and
"worse" count only fields with a direction — hits, IoUs and recalls up, contract and validity booleans true
(`direction-0344.py`). Both scripts are in the run directory.

#### GGUF, think off: 210 cells moved, and reverting `ce8caa6e6` restores production on both probes

| model | cells | control vs production's recorded run (`prod0342a_`, 2026-09-20) | **candidate vs control** | better | worse |
|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | 868 | **0** | **30** | 16 | 7 |
| gemma4:26b-a4b-it-q4_K_M | 862 | **0** | **28** | 5 | 14 |
| gemma4:e4b-it-q4_K_M | 863 / 864 | **0** | **53** | 26 | 12 |
| gemma4:e2b-it-q4_K_M | 866 | **0** | **49** | 12 | 16 |
| qwen3.8:27b-q4_K_M | 858 | **0** | **20** | 10 | 4 |
| qwen3.6:35b-a3b-q4_K_M | 860 | **0** | **30** | 11 | 9 |
| nemotron3:33b-q4_K_M | 864 | **0** | **0** | – | – |
| nemotron3:33b-q8 | 867 | **0** | **0** | – | – |

The first column makes the second a measurement: a fresh control on production's image reproduces production's
five-day-old run in all 6,908 cells, so GGUF think-off on this host is deterministic across days, containers and GPU
contention. The candidate moves 210 of 6,909, in both directions: 80 better, 62 worse, and 68 in fields without a
direction, such as lengths and token counts. Every moved cell is in a model whose attention uses head dimension 256
(gemma4 also 512); nemotron3, at 128, moved in none.

One commit in `b10969..b11081` retunes FlashAttention for exactly those head dimensions: `ce8caa6e6`, *CUDA: tune FA
for Gemma 4 on Ampere or newer*. Its host half changes the Ada decode-kernel selection in `fattn.cu`; its device half
changes the MMA configuration table and tile sizes for D = 256 and 512 in `fattn-mma-f16.cuh`. Built for `120-virtual`
with only that commit reverted, the library's PTX differs from the shipped one in 109 of 6,214 kernels, all
`flash_attn_ext_f16`, none at D = 128. It is NVIDIA-only; on gfx1151 it compiles to a template-parameter rename, and
that host's gate 6 moved nothing.

**Reverting that one commit returns the candidate to production, cell for cell.** The candidate image ran with one
file swapped: `cuda_v13/libggml-cuda.so` rebuilt from b11081 and the fork's series with `ce8caa6e6` reverted
(`977acbeed7ffed64`, checked inside the container). Same suite, same environment, beside the same other workload:

| model | cells | reverted vs production (`ctl0344a_`) | reverted vs the fold (`sync0344a_`) | the fold vs production |
|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | 868 | **0** | 30 | 30 |
| qwen3.6:35b-a3b-q4_K_M | 860 | **0** | 30 | 30 |

On both probes `ce8caa6e6` is the whole of b11081's movement: every moved cell comes back, and nothing else moves. The
probes cover head dimensions 512 and 256 (gemma4:31b) and 256 with 8-way GQA (qwen3.6). The other four models that
moved (gemma4:26b-a4b, e4b and e2b, and qwen3.8, 150 cells) share head dimension 256 and were not re-run, so for them
this is attribution by mechanism, not by measurement.

**The two halves split by model.** The same test with only the device half reverted (`6dd809df497a6a4e`):

| model | cells | device half reverted vs production | vs the fold | vs the full revert |
|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | 868 | **0** | 30 | 0 |
| qwen3.6:35b-a3b-q4_K_M | 860 | 30 | **0** | 30 |

gemma4:31b's movement is the device half, the compile-time tiling. qwen3.6's is the host half: the Ada decode-kernel
selection in `fattn.cu` fires on sm_120 for its shape, and reverting only the tiling leaves every one of its 30 moved
cells where the fold put them.

**On the image built with 908** (`90bb7ffc0be6`), preflight passes (run 3: PASS=21 SKIP=8, every value as run 2),
and the whole GGUF think-off suite ran against production's control and the unpatched fold:

| model | cells | 908 image vs production | vs the fold | moves with |
|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | 868 | **0** | 30 | the device half |
| gemma4:26b-a4b-it-q4_K_M | 862 | **0** | 28 | the device half |
| gemma4:e4b-it-q4_K_M | 863 / 864 | **0** | 53 | the device half |
| gemma4:e2b-it-q4_K_M | 866 | 62 | 54 | both halves |
| qwen3.8:27b-q4_K_M | 858 | 17 | 7 | mostly the host half |
| qwen3.6:35b-a3b-q4_K_M | 860 | 30 | **0** | the host half |
| nemotron3:33b-q4_K_M | 864 | **0** | **0** | neither |
| nemotron3:33b-q8 | 867 | **0** | **0** | neither |

gemma4:e2b is the one gemma4 that 908 does not return to production, and which part of the host half reaches it is
not isolated. Its moved scored fields go 23 better and 13 worse, and its headline reads better than either build:
scene IoU 0.466 against production's 0.300 and the fold's 0.259, and 4/6 boxes against 3/6
(`summarize_engine_compare.py`). An earlier version of this record said the device half accounts for every gemma4 cell;
that held for 31b, 26b and e4b only.

**This run's decode rates are the host's, not the build's.** Every model decoded 2–5× slower than production's
control, nemotron3 included (66 against 281 tok/s), though 908 does not touch its kernels. GPU telemetry, logged
from this run on, shows why: 11–13% utilisation at full clocks and 132–145 W, while the host's load average read 72–85
on its 32 cores. The paired loop-rate arms, which shared one condition, put the device half level with or ahead of the
tiling (gemma4:26b 176 against 137 tok/s, 31b 57 against 57).

The candidate as rendered by `summarize_engine_compare.py` is below: the tables and provenance line verbatim, its two
headings demoted to sit inside this section. Against the control's tables, six quality cells differ: the scene IoU of
four models (gemma4:31b 0.963 → 0.966, e4b 0.461 → 0.458, e2b 0.300 → 0.259, qwen3.6 0.972 → 0.976), e2b's 12 px
tier (3 → 4) and e4b's answer length (480 → 418 tokens).

##### Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | 8192 | 0.966 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 8192 | 0.977 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:e4b-it-q4_K_M | GGUF | 8192 | 0.458 | 4/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:e2b-it-q4_K_M | GGUF | 8192 | 0.259 | 3/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| qwen3.8:27b-q4_K_M | GGUF | 8192 | 0.977 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 8192 | 0.976 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| nemotron3:33b-q4_K_M | GGUF | 8192 | 0.862 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| nemotron3:33b-q8 | GGUF | 8192 | 0.870 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 4/5 |

##### Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 538 | 18 | 607 | 33.3 | 108 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 536 | 52 | 1680 | 11.4 | 317 |
| gemma4:e4b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 3 | 1 | 1 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 418 | 49 | 6915 | 8.8 | 407 |
| gemma4:e2b-it-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 1 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 546 | 258 | 30128 | 2.2 | 1659 |
| qwen3.8:27b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 2 | 1 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 544 | 70 | 1774 | 9.2 | 390 |
| qwen3.6:35b-a3b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 2 | 2 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 550 | 111 | 3291 | 5.8 | 626 |
| nemotron3:33b-q4_K_M | GGUF | 8192 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 284 | 5010 | 2.3 | 1542 |
| nemotron3:33b-q8 | GGUF | 8192 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 242 | 5314 | 2.6 | 1373 |

Provenance (from score files): host(s) http://127.0.0.1:11532 · build(s) 0.34.3-dynres-5-g29ae523 · think=false

**The throughput columns are not comparable across arms.** The candidate's gemma4:31b, 26b and e4b ran while the other
workload was busy, and decoded at a quarter to a third of the control's rate, evenly across all 27 blocks, rising
again over e4b's last seven. Under the same workload today, the ce8caa6e6-reverted build decodes gemma4:31b at
the same 17–19 tok/s, so the slowdown is contention, not the build.

#### MLX, think off: overlapping MLX-CUDA's own spread, and no consistent difference

MLX on CUDA is not bit-reproducible from run to run, so each arm ran twice, interleaved, and the verdict is against
the spread rather than against zero.

| model | cells | production vs itself (2026-09-20) | control vs production's two recorded runs | control vs itself | candidate vs itself | **candidate vs control**, all pairings |
|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | 864 | 13 | 4 – 19 | 19 | 21 | **11 – 28** |
| gemma4:26b-nvfp4 | 861 | 14 | 26 – 33 | 36 | 22 | **26 – 32** |
| gemma4:31b-nvfp4-tower4bit | 867 | 19 | 23 – 31 | 26 | 39 | **30 – 44** |
| qwen3.8:27b-nvfp4 | 862 | 1 | 9 – 11 | 19 | 2 | **10 – 12** |
| qwen3.6:35b-a3b-nvfp4 | 860 | 5 | 6 – 30 | 34 | 47 | **31 – 45** |

qwen3.8's second control run lost 16 blocks when MLX's admission check refused its load beside the other workload
(28.1 GiB needed, 26.2 GiB free); those blocks were re-run on their own.

Against the same-build pairs (production against itself, the control against production's recorded runs, and each
arm against itself), the candidate-vs-control counts overlap on every model: 1–19 against 10–12 on qwen3.8, 5–47
against 31–45 on qwen3.6, 14–36 against 26–32 on gemma4:26b. They reach past the top of the same-build range on two,
gemma4:12b (4–21 against 11–28) and gemma4:31b-nvfp4-tower4bit (19–39 against 30–44), and 31b's cross counts sit
highest overall. A scan of every cell for a value that all five runs not on the candidate share and both candidate
runs replace with another finds one in 4,513, and it does not survive repeats (below).

**That one is a fine-text item, and it is noise.** It is gemma4:31b-nvfp4-tower4bit's 7 px tier (/4), as the
standalone probe reads it (`finetext_probe.py`, the `ft_` file). That is not the reading the generator renders: the
suite runs the same prompt as its own last arm, and `summarize_engine_compare.py` takes the tiers from that block,
falling back to `ft_` only for runs older than the fold that added it. Both readings, every run of this model, and
three more probe-only runs per arm, interleaved, each in a cold container:

| run | build | suite `finetext` block, 7 px | probe `ft_`, 7 px |
|---|---|---|---|
| production, recorded 2026-09-20 (×2) | `0.34.2-dynres-0-g5bffaac` | 3, 3 | 3, 3 |
| the 0.34.2 fold's candidate | `0.34.1-dynres-26-g3dade56` | 3 | 3 |
| control, gate 6 (×2) | `0.34.2-dynres-0-g5bffaac` | **2**, 3 | 3, 3 |
| candidate, gate 6 (×2) | `0.34.3-dynres-5-g29ae523` | **2**, 3 | **2**, **2** |
| control, probe-only (×3) | `0.34.2-dynres-0-g5bffaac` | 3, **2**, 3 | 3, **2**, 3 |
| candidate, probe-only (×3) | `0.34.3-dynres-5-g29ae523` | 3, 3, **2** | 3, 3, 3 |

The generator's reading gives 2 in two of seven runs on production's image and two of five on the candidate's. The
tier flips on both builds, and the other tiers (4/4/4/3) never move. An earlier version of this section, and a comment
on #375, called this a consistent difference from the probe's column alone; that was wrong.

#### OCRBench: production's scores on three arms, one reproducible item on GGUF q4

Rows 0–200 (four of ten categories, not an OCRBench score), gemma4:31b, think off, on the four arms production was
measured on, n = 2 each against production's recorded runs (2026-09-20), rendered by `summarize_extbench.py
--repeats --categories` verbatim. The MIXED banner is correct: two builds is the point of the table.

| model | scored | errors | empty | correct | accuracy | think | endpoint |
|---|---|---|---|---|---|---|---|
| `gemma4:31b-nvfp4-tower4bit` | 200 | 0 | 0 | 173 | **0.865** | false | generate |
| `gemma4:31b-nvfp4-tower4bit` | 200 | 0 | 0 | 173 | **0.865** | false | generate |
| `gemma4:31b-nvfp4` | 200 | 0 | 0 | 171 | **0.855** | false | generate |
| `gemma4:31b-nvfp4` | 200 | 0 | 0 | 171 | **0.855** | false | generate |
| `gemma4:31b-it-q4_K_M` | 200 | 0 | 0 | 171 | **0.855** | false | generate |
| `gemma4:31b-it-q4_K_M` | 200 | 0 | 0 | 170 | **0.85** | false | generate |
| `gemma4:31b-it-q8_0` | 200 | 0 | 0 | 170 | **0.85** | false | generate |
| `gemma4:31b-it-q8_0` | 200 | 0 | 0 | 170 | **0.85** | false | generate |

ocrbench — `echo840/OCRBench` [test], rows 0..200.

⚠ **MIXED — rows are not one campaign** (hosts: ['http://127.0.0.1:11543', 'http://127.0.0.1:11544', 'http://127.0.0.1:11545']; builds: ['0.34.2-dynres-0-g5bffaac', '0.34.3-dynres-5-g29ae523'])

| arm | runs | accuracies | items that changed verdict |
|---|---|---|---|
| prod-nvfp4t | 2 | 0.865, 0.860 | 1 |
| cand-nvfp4t | 2 | 0.865, 0.865 | 0 |
| prod-nvfp4lib | 2 | 0.855, 0.855 | 0 |
| cand-nvfp4lib | 2 | 0.855, 0.855 | 0 |
| prod-q4 | 2 | 0.855, 0.855 | 0 |
| cand-q4 | 2 | 0.850, 0.850 | 0 |
| prod-q8 | 2 | 0.850, 0.850 | 0 |
| cand-q8 | 2 | 0.850, 0.850 | 0 |

| question type | n | prod-nvfp4t | cand-nvfp4t | prod-nvfp4lib | cand-nvfp4lib | prod-q4 | cand-q4 | prod-q8 | cand-q8 |
|---|---|---|---|---|---|---|---|---|---|
| Artistic Text Recognition | 50 | 49/50 | 49/50 | 49/50 | 49/50 | 49/50 | 48/50 | 48/50 | 48/50 |
| Handwriting Recognition | 50 | 34/50 | 34/50 | 34/50 | 34/50 | 33/50 | 33/50 | 33/50 | 33/50 |
| Irregular Text Recognition | 50 | 40/50 | 40/50 | 39/50 | 39/50 | 40/50 | 40/50 | 40/50 | 40/50 |
| Regular Text Recognition | 50 | 50/50 | 50/50 | 49/50 | 49/50 | 49/50 | 49/50 | 49/50 | 49/50 |

- **Both MLX arms and GGUF q8 equal production.** The candidate's tower4bit runs agree with each other and with
  production's first run in every item; production's own two runs differ by one.
- **GGUF q4 loses one Artistic Text item, deterministically.** Both candidate runs miss it and both production runs
  get it; paired, 170 both right, 29 both wrong, 1 production-only, McNemar exact p = 1.000. It is b11081's GGUF
  movement at the size of one item, the same kind as the think-off cells above.
- **The item is `ce8caa6e6`'s device half.** The same arm on the candidate image with only the device half reverted
  (`6dd809df497a6a4e`, n = 1, the arm being deterministic) scores 171, the same as production in every item.
  `summarize_extbench.py --paired`, verbatim:

  | pair | both ✓ | both ✗ | A only | B only | McNemar exact p |
  |---|---|---|---|---|---|
  | prodocr_q4_r1 vs c0344ocr_q4_r1 | 170 | 29 | 1 | 0 | 1.000 |
  | prodocr_q4_r1 vs c0344ocr_q4dev_r1 | 171 | 29 | 0 | 0 | 1.000 |
  | c0344ocr_q4_r1 vs c0344ocr_q4dev_r1 | 170 | 29 | 0 | 1 | 1.000 |

#### Think on: a CUDA-only loop from ce8caa6e6's device half

The three hosts run one protocol (#375): the full ladder 16384 → 131072, the fold's single pass against two-pass
(`OLLAMA_FORMAT_TWO_PASS=1`) on the same image, and the thinking's repetition profile (lines, distinct lines, top
repeat) to tell a loop from long reasoning.

On CUDA GGUF, gemma4:26b-a4b-it-q4_K_M `multi_3img_anchored` does not converge on the fold, in either flow: at the
131072 rung it spends all 122,880 tokens thinking, 68 distinct lines of 4,152, one repeated 1,021 times. Production
finishes the same request. gfx1151 runs the same b11081 and finishes it.

| build | `multi_3img_anchored`, think on |
|---|---|
| production's image (b10969) | stops at 3,883 tokens, valid JSON |
| the candidate (b11081), single pass and two-pass | not converged at 131072 |
| the candidate, `libggml-cuda.so` with ce8caa6e6 reverted | byte-identical to production: 3,883 tokens, same thinking, same answer |
| the same, only the host half reverted | the candidate's loop, byte-identical at the 32768 rung |
| the same, only the device half reverted | byte-identical to production |

The loop is the compile-time tiling, so no runtime switch can select the old behaviour for it. **The maintainer's
decision (2026-09-26): carry the device half as compat patch 908** (`llama/compat/908-revert-fattn-mma-gemma4-tiling.patch`,
124 lines in one CUDA source file). It removes the loop and restores the think-off cells of gemma4:31b, 26b and e4b;
gemma4:e2b and the qwen models also move with the host half, which stays upstream's (above).

**Loop rates.** The full think-on suite on the full ladder, single pass, the fold against the device-half revert on
the same image with the same suite order, so the same history (GGUF loops here are history-independent). Single runs,
because CUDA GGUF is deterministic; gfx1151's column is the ROCm host's, on the same b11081, where the tiling does not
apply.

| model | build | NOT CONVERGED at 131072 | `stop` | `json_valid` | `contract_followed` |
|---|---|---|---|---|---|
| gemma4:26b-a4b-it-q4_K_M | CUDA, the fold as shipped | **6** | 21 / 27 | 21 / 27 | 15 / 20 |
| | CUDA, device half reverted | **1** | 26 / 27 | 25 / 27 | 16 / 20 |
| | gfx1151, the fold (both flows) | **1** | 26 / 27 | 26 / 27 | — |
| gemma4:31b-it-q4_K_M | CUDA, the fold as shipped | 0 | 27 / 27 | 27 / 27 | 18 / 20 |
| | CUDA, device half reverted | 0 | 27 / 27 | 27 / 27 | 18 / 20 |

- **On gemma4:26b the tiling adds five loops that never end.** All six of the fold's are verbatim repetition (68–177
  distinct lines of 4,152–7,171, the top line ×353–1,021): `multi_3img_anchored`, `scene_single`, `bboxm_pin_anc_pos`,
  `bboxm_free_anc_named`, `bbox_contract_positional_1img` and `bbox_contract_adv_real`. Every one finishes on the
  device-half build, at 16384 or 32768, with ordinary thinking. That build's single loop, `bbox_contract_box2d_1img`,
  is a case the fold finished; which case loops moves with the numerics, and the count matches gfx1151's.
- **On gemma4:31b it changes numerics, not outcomes.** Nothing caps on either build; 55 of 867 cells differ, small IoU
  shifts in both directions.
- The arms did not alternate their order as the script intended (a loop counter shared with the server-wait loop);
  GGUF is deterministic, so the order does not change a result.

The other hosts' loops are different cases. The MLX single-pass loop the Metal host found on gemma4:31b-nvfp4 does
not reproduce on MLX-CUDA: no loop in four runs, and here the spread within one flow is as large as the spread between
flows. On gemma4:26b-nvfp4, Metal's single pass leaves five more cases looping than its two-pass (interim), and Metal's
discriminator puts that on drafting rather than on the flow. gfx1151 has one pre-existing gemma4:26b GGUF loop,
`bbox_contract_real_1img`, byte-identical on b10969. On CUDA, the loop-rate run covers ROCm's case, and the MLX
think-on run covers Metal's: the two cases Metal found on 31b, on all four models, plus Metal's five 26b cases on
gemma4:26b-nvfp4, both flows, two repeats each, the full ladder.

Queued, in order: the rest of gate 6's MLX controls, gate 5 run 2, the fine-text repeats, OCRBench, the loop rates,
MLX think-on, and a drafting probe (the knob and the flow, apart).

#### MLX drafting probe: the knob decides whether think+format drafts, and drafting is 1.5–1.7× on the thinking

This measures item 5 of "What single pass changes". In the single pass a think+format request reaches the MLX runner
with its grammar attached from the first token: `requestGrammar` (`mlxrunner/client.go`) puts the schema behind a
free-thinking `any_text` element. `draftingEnabled` (`mlxrunner/speculate.go`) decides once per request, on
`request.Grammar == nil || draftUnderGrammar`. So under production's `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0` the single
pass drafts nothing, while the two-pass flow's first pass, which carries no grammar, drafts its thinking.

Three arms on the fold's image, each in a canary container with production's environment otherwise, from 2026-09-26
14:43 to 2026-09-27 01:18:

| arm | flow | knob | drafts the thinking | stands for |
|---|---|---|---|---|
| F0 | single pass | 0 | no | the fold, deployed with production's environment |
| P0 | two-pass (`OLLAMA_FORMAT_TWO_PASS=1`) | 0 | yes, in pass one | production today |
| F1 | single pass | 1 | yes, under the grammar | the knob turned on |

Each arm sent one text-only think+format request: a routing puzzle, a three-field JSON schema, temperature 0, seed 42,
`num_ctx` 16384, `num_predict` 8192. It was sent three times to warm up and five times measured. Each model ran three
rotations, with the arm order rotating. Every request in every arm thought until the 8,192-token cap, so this measures
thinking speed and says nothing about answers.

The windows were not quiet. The host's load reached 78 during qwen3.8's first rotation. From 20:07 other jobs shared
the GPU: short bursts at first, then a steady one from 21:57 that grew from about 11 % of the SMs to about 40 %. Each
request's window, taken from the runner log, was read against 5-second telemetry: every process's SM share from
`nvidia-smi pmon`, and the load average. The listing gives the median of the five measured requests per arm and
rotation. Beside it are the share of the window in which other processes held at least 10 % of the SMs (`cont`), their
mean SM % (`peerSM`), and the mean load:

```
qwen3.8:27b-nvfp4
F0 rot 1: n=5 median  17.10  range 12.9-22.4  cont   0%  peerSM   0.0  load1  46.7
F0 rot 2: n=5 median  26.90  range 23.1-28.4  cont   0%  peerSM   0.0  load1   8.7
F0 rot 3: n=5 median  26.21  range 20.9-28.0  cont   0%  peerSM   0.0  load1  12.4
F1 rot 1: n=5 median  42.09  range 36.4-55.7  cont   0%  peerSM   0.0  load1  26.2
F1 rot 2: n=5 median  45.43  range 37.3-53.1  cont   0%  peerSM   0.0  load1  12.8
F1 rot 3: n=5 median  57.74  range 47.7-58.5  cont   0%  peerSM   0.0  load1   8.2
P0 rot 1: n=5 median  44.12  range 41.1-50.2  cont   0%  peerSM   0.1  load1  19.3
P0 rot 2: n=5 median  42.58  range 40.0-49.4  cont   0%  peerSM   0.0  load1  21.0
P0 rot 3: n=5 median  45.77  range 38.0-47.3  cont  10%  peerSM   3.0  load1  16.9

gemma4:31b-nvfp4
F0 rot 1: n=5 median  31.97  range 27.2-33.0  cont  27%  peerSM  10.6  load1   9.2
F0 rot 2: n=5 median  26.36  range 26.3-26.6  cont  93%  peerSM  31.1  load1  11.4
F0 rot 3: n=5 median  24.48  range 24.3-30.3  cont  74%  peerSM  31.1  load1  14.9
F1 rot 1: n=5 median  54.03  range 50.7-57.5  cont   0%  peerSM   0.0  load1   9.4
F1 rot 2: n=5 median  51.46  range 46.5-53.7  cont  51%  peerSM  11.6  load1  11.2
F1 rot 3: n=5 median  38.79  range 38.6-40.9  cont  97%  peerSM  36.2  load1  14.5
P0 rot 1: n=5 median  48.00  range 46.6-51.5  cont   8%  peerSM   3.5  load1  13.1
P0 rot 2: n=5 median  47.54  range 46.5-51.9  cont  54%  peerSM  11.3  load1  14.7
P0 rot 3: n=5 median  40.51  range 33.9-53.2  cont  96%  peerSM  44.1  load1  11.4
```

- **Drafting is 1.5–1.7× on the thinking, and production's two-pass has it.** Compare production's flow with the
  fold's default under production's knob, P0 against F0:
  - qwen3.8: 44.1 tok/s across P0's fifteen requests, against 26.4 across F0's two quiet rotations (1.7×).
  - gemma4:31b, rotation 1: 48.0 against F0's two uncontended requests, 32.7 and 33.0 (1.46×).
  - gemma4:31b, rotation 3, with every arm under the heavy peer: 40.5 against 24.5 (1.65×).

  F1 over F0 reads 1.6–2.2×.
- **So the flow matters only through the knob.** Deployed with production's environment, the fold's default thinks
  1.5–1.7× slower on MLX than production does today. So the deploy runs two-pass (open item 7, the maintainer's
  decision).
- **F1 and P0 are level within the spread.** Both draft the thinking. On gemma4:31b, F1 over P0 reads 1.13, 1.08 and
  0.96 across the rotations; on qwen3.8 it reads 0.95, 1.07 and 1.26. Within one arm, speed follows the request's draft
  acceptance. In P0's third rotation on 31b, under the same peer, one request ran at 53.2 tok/s with acceptance 0.84
  and the next at 33.9 with 0.50.
- **Drafting changes greedy output on MLX-CUDA.** With drafting off, requests 2–8 of an arm repeat their thinking
  exactly: gemma4:31b gives 15,386 characters every time, qwen3.8 gives 13,151, and the first request after a load
  differs. Every drafted request thinks differently: 12,894–16,398 characters on 31b, 11,151–16,201 on qwen3.8. That is
  why acceptance moves from request to request. It is the same effect as the think-off flips that warm drafting adds.

#### KV precision and flash attention against the gemma4:26b GGUF loops (#387)

These are #387's cold captures on both builds. The cases are `multi_3img_anchored`, `bbox_contract_real_1img` and
`bbox_contract_box2d_1img`. The arms are f16 and f32, each with flash attention on and off. The run was 2026-09-27
01:18–04:29, and the full table is on #387.

- **Two byte-identities.** With flash attention on, f32 reproduces f16 byte for byte on both builds: CUDA's flash
  attention converts an f32 cache to f16 first. With flash attention off, the fold and the 908 image are
  byte-identical at both precisions, because 908 changes only the flash-attention tiling.
- **What loops.** The flash-attention-on columns repeat the loop-rate run. The tiling loops `multi_3img_anchored`,
  which the 908 image finishes in 3,882 tokens against production's 3,883. The revert loops `box2d_1img`. `real_1img`
  loops on both builds.
- **Flash attention off.** At f16 it finishes all three cases. At f32 it loops `multi_3img_anchored`.
- **The verdict.** No KV type or attention path reliably removes these loops, which matches the gfx1151 host's
  reading. Production keeps f16 with flash attention on (ADR 0043).

#### MLX think-on with a fixed history: the same counts, and one stable loop

Metal found on #375 that MLX's think-on loops depend on request history. A gemma4:26b case that looped twice in the
suite finished as the first request after a cold restart. The agreed protocol's run (above) sends a model's cases in
sequence on one server, so its per-flow counts carry that history.

This variant fixes the history. It sends every case alone, and the suite restarts the server before every rung
(`RESTART_CMD`), so every rung of every case is the first request after a cold load. Everything else is the same:
image, cases, flows, environment (`OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`), the ladder 16384 → 131072, and two repeats. It
ran 2026-09-27 04:29–17:40, with a pause from 11:12 to 11:35; the interrupted arm was re-run whole.

`fh-compare-0344.py` renders both runs from the suite's score files: the history-laden run (`on0344*`, "hist") beside
the fixed-history run (`fh0344*`, "fixed"). In the column names, `f` is the single pass, `p` is two-pass, and the digit
is the repeat. A cell is `NC` (not converged at 131072), or the thinking's token count and the rung it finished on:

| model | case | hist f1 | hist f2 | hist p1 | hist p2 | fixed f1 | fixed f2 | fixed p1 | fixed p2 |
|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-nvfp4 | `scene_single_pinned` | 2012·16k | 2007·16k | 2012·16k | 2011·16k | 2007·16k | 2012·16k | 2009·16k | 2009·16k |
| gemma4:31b-nvfp4 | `multi_3img_anchored` | 5749·16k | 6272·16k | 6493·128k | 6459·16k | 5749·16k | 6698·16k | 7109·16k | 5924·32k |
| qwen3.8:27b-nvfp4 | `scene_single_pinned` | 1480·16k | 2065·16k | 1502·16k | 1566·16k | 1641·16k | 1429·16k | 1696·16k | 1787·16k |
| qwen3.8:27b-nvfp4 | `multi_3img_anchored` | 1804·16k | 1934·16k | 1781·16k | 1711·16k | 1799·16k | 1794·16k | 1799·16k | 1679·16k |
| gemma4:26b-nvfp4 | `scene_single_pinned` | 6367·16k | 5535·16k | 6568·16k | 3614·16k | 3687·16k | 4195·16k | 6419·16k | 4003·16k |
| gemma4:26b-nvfp4 | `multi_3img_anchored` | **NC** | **NC** | **NC** | **NC** | **NC** | **NC** | **NC** | **NC** |
| gemma4:26b-nvfp4 | `bboxm_free_noanc_pos` | 7314·16k | 2720·32k | 3129·16k | 7395·32k | 2864·16k | 5372·16k | 5199·32k | 4319·32k |
| gemma4:26b-nvfp4 | `scene_single_anchored` | 7062·16k | 4987·16k | 5800·16k | 4564·16k | 8881·32k | 7594·16k | 5391·16k | 7635·16k |
| gemma4:26b-nvfp4 | `multi_3img` | 6658·16k | 7658·16k | 8253·16k | 6900·64k | 8431·64k | 5948·32k | 7582·128k | 6371·32k |
| gemma4:26b-nvfp4 | `bbox_contract_box2d_1img` | 5777·16k | 4003·16k | 2614·16k | 2620·16k | 4027·16k | 6079·16k | 2602·16k | 2696·16k |
| gemma4:26b-nvfp4 | `bbox_contract_positional_1img` | 5321·16k | 2683·32k | 3815·32k | 3892·16k | 3747·16k | 1770·16k | 2937·16k | 5120·16k |
| qwen3.6:35b-a3b-nvfp4 | `scene_single_pinned` | 6417·64k | 4145·16k | 19941·64k | 13610·128k | 6702·16k | 9101·32k | 16622·128k | **NC** |
| qwen3.6:35b-a3b-nvfp4 | `multi_3img_anchored` | 9958·64k | 9451·32k | 12768·32k | 13452·32k | 10185·32k | 12581·64k | 14647·32k | 3525·16k |

```
history-laden single pass repeat 1: NOT CONVERGED 1 of 13 scored cases
history-laden single pass repeat 2: NOT CONVERGED 1 of 13 scored cases
history-laden two-pass repeat 1: NOT CONVERGED 1 of 13 scored cases
history-laden two-pass repeat 2: NOT CONVERGED 1 of 13 scored cases
fixed-history single pass repeat 1: NOT CONVERGED 1 of 13 scored cases
fixed-history single pass repeat 2: NOT CONVERGED 1 of 13 scored cases
fixed-history two-pass repeat 1: NOT CONVERGED 1 of 13 scored cases
fixed-history two-pass repeat 2: NOT CONVERGED 2 of 13 scored cases
```

- **Fixing the history changes no count.** Every arm leaves 1 of 13 cases not converged in both runs. The one
  exception is the fixed-history two-pass repeat 2, with 2.
- **gemma4:26b `multi_3img_anchored` never converges, in any of the 8 runs**: both flows, both histories, both
  repeats. So neither the flow nor the history decides it. Its prompt is `multi_3img`'s plus one calibration
  paragraph, and that paragraph ends with #387's trap sentence: "If you resized image 1 internally, use the size YOU
  used." The source is `vision_suite.py`, `MULTI_ANCHORED_PROMPT = MULTI_PROMPT + …`; the gfx1151 host pointed it out
  on #375. The images and the scorer are the same, so the table holds a minimal pair: `multi_3img` converges in 8 of
  8 runs, `multi_3img_anchored` in 0 of 8. On gfx1151's GGUF under `q8_0` the same paragraph triples the thinking
  (5,236 to 15,788 characters; the whole generation goes from about 3,560 to 8,330 tokens, 2.3×) and leaves the answer
  about as long. Under f16, production's setting, it loops, and replacing the sentence ends the loop (item 8's GGUF
  leg, in [the gfx1151 section](#gate-6-think-on-under-the-aligned-protocol)). The pair isolates the paragraph; item 8
  isolates the sentence.
- **Metal's five 26b cases converge in every run on CUDA**, in both flows and both histories. In the fixed-history run
  some converge only on a higher rung. Each rung there is a fresh cold load, so that is MLX-CUDA's per-load variation,
  not history.
- **One case leans against the drafted two-pass flow: qwen3.6 `scene_single_pinned`.** In both histories, every
  two-pass run thinks longer than every single-pass run. Two-pass took 13,610 to 19,941 tokens, 16,622 on the top rung,
  and has the one extra NOT CONVERGED; single pass took 4,145 to 9,101. Four runs against four on one case is a lean,
  not a verdict. It points the same way as Metal's discriminator: at drafting, not the flow.
- **So on MLX-CUDA think-on, the single-pass default loops no more than two-pass, with or without a fixed history.**

#### Open item 8 on CUDA: the trap sentence

Item 8 asks whether the trap sentence alone, "If you resized image 1 internally, use the size YOU used.", makes
gemma4:26b loop on `multi_3img_anchored`. `promptcap.py` (#387, `b13f2c35d`) sends the suite's own request
(`vision_suite.gen`, format `"json"`), cold, greedy, at 32768 (24,576 tokens), in three variants. `orig` is unchanged.
`size` replaces the sentence with image 1's size (1920×1080). `commit` replaces it with an instruction to commit to one
estimate. `multi_3img`'s `orig` is the control. It ran on 2026-09-27, one container at a time on GPU0:

- **GGUF**: gemma4:26b-a4b-it-q4_K_M with production's settings (f16 KV, flash attention on), one capture each, on the
  fold image (`sync-0.34.4`) and on the image that ships (`sync-0.34.4-908`). CUDA GGUF is deterministic. The fold's
  `orig` capture is a byte-exact prefix of #387's capture of the same prompt on the fold, f16 with flash attention
  on, at 65536: all 61,602 characters of its thinking match.
- **MLX**: gemma4:26b-nvfp4 in the fixed-history run's container (production's environment, single pass,
  `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`), five cold captures of each prompt, the order rotated each repeat. With the knob
  at 0 the single pass never drafts. Even so, every cold MLX-CUDA load takes its own greedy trajectory, and the runner
  logs show one runner start per capture. So each capture is one draw, and the answer is a rate.

`item8-cuda-read-0344.py` (in the run directory) reads each capture with #387's `kvloop_read.py` and scores it with
`score_multi`, one line per capture, as the gfx1151 host's `item8_read.py` does. GGUF, verbatim:

```
$ item8-cuda-read-0344.py --compact promptcap-cuda        # the fold image
== gguf: gemma4:26b-a4b-it-q4_K_M, server 0.34.3-dynres-5-g29ae523
multi_3img orig: prompt 5e3ea981f6d5551e images 7a5285fe1380ac49 num_ctx 32768 num_predict 24576
  r1: stop    5,425 | 2nd half 90/90 | x3 '- **Key Objects:**' | no loop | all right, anchor None
  finished 1 of 1, every question right in 1
multi_3img_anchored orig: prompt aa1042593bd9fc5c images 7a5285fe1380ac49 num_ctx 32768 num_predict 24576
  r1: length 24,576 | 2nd half 9/437 | x203 '- ANCHOR: [72, 148, 216, 336]' | loop from ~2,129 | no answer
  finished 0 of 1, every question right in 0; of the 1 capped, the most repeated line is a box in 1
multi_3img_anchored size: prompt 5e75102279f8d97e images 7a5285fe1380ac49 num_ctx 32768 num_predict 24576
  r1: stop    6,595 | 2nd half 143/147 | x3 '- Key objects:' | no loop | all right, anchor [0, 0, 1920, 1080]
  finished 1 of 1, every question right in 1
multi_3img_anchored commit: prompt f1b94fb7c51eec6e images 7a5285fe1380ac49 num_ctx 32768 num_predict 24576
  r1: stop    5,863 | 2nd half 105/108 | x3 '- Key objects:' | no loop | all right, anchor [0, 0, 1920, 1080]
  finished 1 of 1, every question right in 1
$ item8-cuda-read-0344.py --compact promptcap-cuda-908    # the image that ships
== gguf: gemma4:26b-a4b-it-q4_K_M, server 0.34.3-dynres-22-g5584539
multi_3img orig: prompt 5e3ea981f6d5551e images 7a5285fe1380ac49 num_ctx 32768 num_predict 24576
  r1: stop    4,692 | 2nd half 80/81 | x3 '- **Key Objects:**' | no loop | all right, anchor None
  finished 1 of 1, every question right in 1
multi_3img_anchored orig: prompt aa1042593bd9fc5c images 7a5285fe1380ac49 num_ctx 32768 num_predict 24576
  r1: stop    3,882 | 2nd half 51/51 | x3 '- **Key objects:**' | no loop | all right, anchor [0, 0, 1000, 562]
  finished 1 of 1, every question right in 1
multi_3img_anchored size: prompt 5e75102279f8d97e images 7a5285fe1380ac49 num_ctx 32768 num_predict 24576
  r1: stop    6,012 | 2nd half 104/104 | x3 '- Key objects:' | no loop | all right, anchor [0, 0, 1920, 1080]
  finished 1 of 1, every question right in 1
multi_3img_anchored commit: prompt f1b94fb7c51eec6e images 7a5285fe1380ac49 num_ctx 32768 num_predict 24576
  r1: stop    4,657 | 2nd half 80/82 | x3 '- Key objects:' | no loop | all right, anchor [0, 0, 1920, 1080]
  finished 1 of 1, every question right in 1
```

For MLX, the same run's output continues. It also counts the fixed-history run's draws from the suite's score files:
each rung there is a cold load, and a ladder escalates only on a cap.

```
== mlx: gemma4:26b-nvfp4, server 0.34.3-dynres-5-g29ae523
multi_3img orig: prompt 5e3ea981f6d5551e images 7a5285fe1380ac49 num_ctx 32768 num_predict 24576
  r1: stop    5,623 | 2nd half 98/135 | x3 '- Text found:' | no loop | all right, anchor None
  r2: stop    9,722 | 2nd half 158/196 | x15 'CIPHER: [600, 166, 781, 391]' | no loop | all right, anchor None
  r3: stop    6,759 | 2nd half 115/137 | x11 '- ANCHOR: [73, 147, 218, 336]' | no loop | all right, anchor None
  r4: length 24,576 | 2nd half 23/598 | x60 '- ANCHOR: [74, 147, 218, 336]' | loop from ~7,676 | no answer
  r5: length 24,576 | 2nd half 4/686 | x545 "Let's try:" | loop from ~8,283 | no answer
  finished 3 of 5, every question right in 3; of the 2 capped, the most repeated line is a box in 1
multi_3img_anchored orig: prompt aa1042593bd9fc5c images 7a5285fe1380ac49 num_ctx 32768 num_predict 24576
  r1: stop   11,342 | 2nd half 110/253 | x22 '- BEACON: [320, 110, 465, 305]' | no loop | all right, anchor [0, 0, 1000, 1000]
  r2: length 24,576 | 2nd half 5/515 | x195 '- BEACON: [320, 111, 465, 305]' | loop from ~2,761 | no answer
  r3: length 24,576 | 2nd half 9/614 | x147 '- ANCHOR: [73, 147, 218, 335]' | loop from ~3,121 | no answer
  r4: length 24,576 | 2nd half 6/516 | x202 '- ANCHOR: [73, 147, 218, 336]' | loop from ~1,457 | no answer
  r5: length 24,576 | 2nd half 5/463 | x219 '- ANCHOR: [73, 147, 218, 336]' | loop from ~3,288 | no answer
  finished 1 of 5, every question right in 1; of the 4 capped, the most repeated line is a box in 4
multi_3img_anchored size: prompt 5e75102279f8d97e images 7a5285fe1380ac49 num_ctx 32768 num_predict 24576
  r1: length 24,576 | 2nd half 7/1116 | x411 '"q1" in image 3.' | loop from ~4,798 | no answer
  r2: stop   10,277 | 2nd half 158/217 | x5 '"MAXUS INDUSTRIAL SUPPLY"' | no loop | all right, anchor [0, 0, 1920, 1080]
  r3: length 24,576 | 2nd half 49/1525 | x59 "Let's check if any word from image" | loop from ~3,248 | no answer
  r4: length 24,576 | 2nd half 11/565 | x122 '- BEACON: [320, 108, 467, 305]' | loop from ~7,309 | no answer
  r5: length 24,576 | 2nd half 17/798 | x91 "Let's re-estimate." | loop from ~5,638 | no answer
  finished 1 of 5, every question right in 1; of the 4 capped, the most repeated line is a box in 1
multi_3img_anchored commit: prompt f1b94fb7c51eec6e images 7a5285fe1380ac49 num_ctx 32768 num_predict 24576
  r1: stop    7,194 | 2nd half 99/182 | x6 '- Key objects:' | no loop | all right, anchor [0, 0, 1000, 1000]
  r2: length 24,576 | 2nd half 6/484 | x187 '- ANCHOR: [73, 147, 218, 335]' | loop from ~1,792 | no answer
  r3: length 24,576 | 2nd half 6/502 | x192 '- ANCHOR: [73, 147, 218, 335]' | loop from ~2,418 | no answer
  r4: length 24,576 | 2nd half 6/515 | x203 '- ANCHOR: [73, 147, 218, 336]' | loop from ~1,530 | no answer
  r5: length 24,576 | 2nd half 6/475 | x186 '- ANCHOR: [73, 147, 218, 335]' | loop from ~1,223 | no answer
  finished 1 of 5, every question right in 1; of the 4 capped, the most repeated line is a box in 4
  per repeat, captures finished: r1 3 of 4, r2 2 of 4, r3 1 of 4, r4 0 of 4, r5 0 of 4
  control finished 3 of 5; the three anchored variants pooled 3 of 15; Fisher two-sided p = 0.131

== suite, fixed-history run (fh0344), gemma4:26b-nvfp4: draws with a known outcome at 24,576 tokens
  multi_3img f1: 16k capped (censored), 32k capped, 64k finished in 8,431
  multi_3img f2: 16k capped (censored), 32k finished in 5,948
  multi_3img single pass: finished within 24,576 in 2 of 3 draws
  multi_3img p1: 16k capped (censored), 32k capped, 64k capped, 128k finished in 7,582
  multi_3img p2: 16k capped (censored), 32k finished in 6,371
  multi_3img two-pass: finished within 24,576 in 2 of 4 draws
  multi_3img_anchored f1: 16k capped (censored), 32k capped, 64k capped, 128k capped
  multi_3img_anchored f2: 16k capped (censored), 32k capped, 64k capped, 128k capped
  multi_3img_anchored single pass: finished within 24,576 in 0 of 6 draws
  multi_3img_anchored p1: 16k capped (censored), 32k capped, 64k capped, 128k capped
  multi_3img_anchored p2: 16k capped (censored), 32k capped, 64k capped, 128k capped
  multi_3img_anchored two-pass: finished within 24,576 in 0 of 6 draws
  single pass, with promptcap's MLX captures: multi_3img 5 of 8, multi_3img_anchored orig 1 of 11; Fisher two-sided p = 0.0408
```

- **GGUF on the fold image: yes, the sentence alone.** This is gfx1151's result. `orig` loops from about token 2,129,
  re-listing image 1's boxes ("- ANCHOR: [72, 148, 216, 336]" ×203, 9 distinct lines in the second half), and never
  answers. `size` and `commit` finish in 6,595 and 5,863 tokens, and the control in 5,425, each with every question
  right.
- **GGUF on the image that ships: no.** On `sync-0.34.4-908` all four finish with every question right. `orig` takes
  3,882 tokens, and its thinking and answer are byte-identical to #387's f16, flash-attention-on capture on the 908
  image at 65536; production finishes the same request in 3,883 (above). `size` takes 6,012, `commit` 4,657 and the
  control 4,692. So on CUDA the loop needs both the sentence and `ce8caa6e6`'s tiling, which 908 reverts (item 2). On
  gfx1151, where that tiling does not apply, the f16 path loops on the sentence by itself.
- **MLX: the sentence does not set the loop rate.** `orig`, `size` and `commit` each finished 1 of 5 draws. The control
  loops too: it finished 3 of 5. Five draws per prompt cannot separate the paragraph from the control: the three
  anchored variants pooled finished 3 of 15 (Fisher's exact test, two-sided, p = 0.13). The fixed-history run's
  single-pass draws lean the same way. There the control finished 2 of 3 and `orig` 0 of 6, which gives 5 of 8
  against 1 of 11 together (p = 0.04).
- **Stating the size changes how the case loops, not how often.**
  - Every capped `orig` and `commit` draw re-lists image 1's boxes, for example "- ANCHOR: [73, 147, 218, 336]"
    ×147–219, from about token 1,200–3,300.
  - With the size stated, the loops start later, at about 3,200–7,300, and they vary. One repeats a box. The others
    repeat a question ('"q1" in image 3.' ×411), a check across the images (×59) or a re-estimate
    ("Let's re-estimate." ×91).
  - The control's two loops start later still, at about 7,700 and 8,300.
- **Finishes fell off by repeat: 3, 2, 1, 0 and 0 of 4.** Every repeat ran each prompt once, in a rotated order, so
  the comparisons above are balanced across repeats. A split chosen after seeing the data is weak evidence of drift,
  and nothing carried over between captures, since each one started a new runner.
- **This is how to read the fixed-history table.** Every rung there is one cold draw. So a case that converges only
  on a higher rung drew a loop first. `multi_3img_anchored`'s four NOT CONVERGED cells in that run are 16 capped
  draws, 12 of them with 24,576 tokens or more.

## Gates 4–6 on gfx1151 (2026-09-25)

**Host.** The ROCm host, `amd-server`: Ryzen AI Max+ 395 with a Radeon 8060S (gfx1151), 96 GiB of VRAM.
- The GPU was shared throughout with production on `:11434`, and during gate 6 with this host's own capture
  containers.
- Production moved to 0.34.3 this morning (#379). No campaign ran on production's container.

### Gate 4: the image

| | candidate | 0.34.3 gate image |
|---|---|---|
| image | `maxusai-ollama:0.34.3-dynres-5-g29ae523-rocm7-gfx1151` | `maxusai-ollama:0.34.2-dynres-24-gef19770-rocm7-gfx1151` |
| built from | `29ae52351`, through `scripts/build_rocm.sh`: `Dockerfile.rocm` on `rocm/dev-ubuntu-24.04:7.2.4-complete`, gfx1151 | the v0.34.3 fold, the same way |
| `llama-server --version` | commit `161755f29` (b11081) | commit `391fac164` (b10969) |
| payload | 1863 entries, 96 gfx1151 rocBLAS kernel files | 1863 and 96. No file on one side only, and no SONAME or symlink change |

- **Patches.** All seven compat patches apply, in both the CPU stage and the HIP stage, and both stages compile.
- **Build time.** The image built in 22 s from ccache. Its native payload is byte-identical to the ROCm cross-check
  build (`dd19f1202`): `llama-server`, `libllama-server-impl.so`, `libggml-hip.so` and `libggml-base.so` have the
  same sha256.
- **The ROCm 10 lane** (experimental, ADR 0040) builds too: 4152 entries and 150 gfx1151 kernel files, the same as
  0.34.3's ROCm 10 image.
- **908 (2026-09-26): no gfx1151 kernel changes.** `558453953` built as
  `maxusai-ollama:0.34.3-dynres-22-g5584539-rocm7-gfx1151` in 47 s from ccache, with all eight patches applied in both
  stages. Against the image above, one payload file of 1863 differs: `libggml-hip.so`.
  - In its gfx1151 code objects, 22 of the 138 offload bundles differ, and only in `.text`. The 78
    `flash_attn_ext_f16` kernels that compile for gfx1151, 18 of them at D = 256, are byte-identical.
  - The other 184 `flash_attn_ext_f16` entries are `NO_DEVICE_CODE` stubs. On RDNA every D = 512 variant is one. Each
    stub differs in one byte, its `__LINE__` literal (1833 → 1807 and 1861 → 1835), because 908 deletes 26 lines above
    them.
  - The library's host code differs in 277 places: 252 `__LINE__` literals of the MMA launcher's two `CUDA_CHECK`s, 18
    in the inlined Ampere rows (the D = 512 constants and the compare chain that selects them), and 7 in alignment
    padding. The host picks the table by device. `ampere_mma_available()` is false on every AMD device, so gfx1151 takes
    the RDNA table, which 908 leaves unchanged.

### Gate 5: preflight

`rocm7-0-34-4-dynres` (#378) was measured on this payload. The result is **VERDICT PASS, PASS=20 SKIP=12**:
- The three ladders reproduce b10864's exactly. `tools/mtmd` changed by one error-handling hunk only.
- `payload_pin` reads `161755f29` through the container route #376 fixed.
- Both pinned budgets hold: 3328 → 3270 and 560 → 529.
- `think_format` passes through the single pass on all three arches, in 1401, 127 and 273 tokens.
- Every skip predates the profile.

**On the 908 image** (`0.34.3-dynres-22-g5584539`), with 908 in the patch set, the verdict is again **PASS, PASS=20
SKIP=12**. Every check reads the same value as on the first image, apart from the version and the image tag: the same
ladders, the same pinned budgets, and `think_format` in 1401, 127 and 273 tokens. The run record is
`runs/preflight-rocm7-0344-fold-g5584539.json`.

### Gate 6: think off and OCRBench

The fold's cells are compared against the 0.34.3 gate image (`r0343cand`) and production 0.34.2 (`r0343ctrl`), with
the same harness and environment. `compare.sh`, verbatim:

```
== think off
  [r0343cand] 0 of 978 cells differ (scores_r0343cand_1_qwen3_6_35b-a3b-q4_k_m_thinkfalse.json vs scores_r0344fold_1_qwen3_6_35b-a3b-q4_k_m_thinkfalse.json)
  [r0343ctrl] 0 of 978 cells differ (scores_r0343ctrl_1_qwen3_6_35b-a3b-q4_k_m_thinkfalse.json vs scores_r0344fold_1_qwen3_6_35b-a3b-q4_k_m_thinkfalse.json)
  [r0343cand] 0 of 976 cells differ (scores_r0343cand_1_qwen3_8_27b-q4_K_M_thinkfalse.json vs scores_r0344fold_1_qwen3_8_27b-q4_K_M_thinkfalse.json)
  [r0343ctrl] 0 of 976 cells differ (scores_r0343ctrl_1_qwen3_8_27b-q4_K_M_thinkfalse.json vs scores_r0344fold_1_qwen3_8_27b-q4_K_M_thinkfalse.json)
  [r0343cand] 0 of 986 cells differ (scores_r0343cand_1_gemma4_31b-it-q4_K_M_thinkfalse.json vs scores_r0344fold_1_gemma4_31b-it-q4_K_M_thinkfalse.json)
  [r0343ctrl] 0 of 986 cells differ (scores_r0343ctrl_1_gemma4_31b-it-q4_K_M_thinkfalse.json vs scores_r0344fold_1_gemma4_31b-it-q4_K_M_thinkfalse.json)
  [r0343cand] 0 of 980 cells differ (scores_r0343cand_1_gemma4_26b-a4b-it-q4_K_M_thinkfalse.json vs scores_r0344fold_1_gemma4_26b-a4b-it-q4_K_M_thinkfalse.json)
  [r0343ctrl] 0 of 980 cells differ (scores_r0343ctrl_1_gemma4_26b-a4b-it-q4_K_M_thinkfalse.json vs scores_r0344fold_1_gemma4_26b-a4b-it-q4_K_M_thinkfalse.json)
  [r0343cand] 0 of 983 cells differ (scores_r0343cand_1_nemotron3_33b-q4_K_M_thinkfalse.json vs scores_r0344fold_1_nemotron3_33b-q4_K_M_thinkfalse.json)
  [r0343ctrl] 0 of 983 cells differ (scores_r0343ctrl_1_nemotron3_33b-q4_K_M_thinkfalse.json vs scores_r0344fold_1_nemotron3_33b-q4_K_M_thinkfalse.json)
```

OCRBench, rows 0–200, gemma4:31b-it-q4_K_M. `summarize_extbench.py --paired --categories`, verbatim:

```
== OCRBench rows 0-200, gemma4:31b-it-q4_K_M
| model | scored | errors | empty | correct | accuracy | think | endpoint |
|---|---|---|---|---|---|---|---|
| `gemma4:31b-it-q4_K_M` | 200 | 0 | 0 | 172 | **0.86** | false | generate |
| `gemma4:31b-it-q4_K_M` | 200 | 0 | 0 | 172 | **0.86** | false | generate |
| `gemma4:31b-it-q4_K_M` | 200 | 0 | 0 | 172 | **0.86** | false | generate |

ocrbench — `echo840/OCRBench` [test], rows 0..200.

⚠ **MIXED — rows are not one campaign** (hosts: ['http://127.0.0.1:11497', 'http://127.0.0.1:11499']; builds: ['0.34.2-dynres-24-gef19770', '0.34.2-dynres-f67b1aef', '0.34.3-dynres-5-g29ae523'])

| pair | both ✓ | both ✗ | A only | B only | McNemar exact p |
|---|---|---|---|---|---|
| r0343ctrl_ocr_q4 vs r0343cand_ocr_q4 | 172 | 28 | 0 | 0 | 1.000 |
| r0343ctrl_ocr_q4 vs r0344fold_ocr_q4 | 172 | 28 | 0 | 0 | 1.000 |
| r0343cand_ocr_q4 vs r0344fold_ocr_q4 | 172 | 28 | 0 | 0 | 1.000 |

| question type | n | prod-0.34.2 | img-0.34.3 | fold-0.34.4 |
|---|---|---|---|---|
| Artistic Text Recognition | 50 | 48/50 | 48/50 | 48/50 |
| Handwriting Recognition | 50 | 34/50 | 34/50 | 34/50 |
| Irregular Text Recognition | 50 | 41/50 | 41/50 | 41/50 |
| Regular Text Recognition | 50 | 49/50 | 49/50 | 49/50 |
```

**b11081 moves no scored think-off cell and no OCRBench item on gfx1151.** That includes three MoE models under
`fccf7166f`, the RDNA3.5 MoE tile heuristic. `ce8caa6e6` does not apply here, because RDNA has its own FA config table.

### Gate 6: think on, under the aligned protocol

**Done 2026-09-27. On GGUF, the single pass loops no more than two-pass on any of the five models.** Each flow leaves
the same cases unfinished: none on gemma4:31b, qwen3.8 and nemotron3; `bbox_contract_real_1img` on gemma4:26b; and
on qwen3.6 two cases under a `q8_0` KV cache and one under f16. Those loops come from the prompt, not the flow.

**Every run here used this host's production KV cache at the time, `q8_0`** (#386), except qwen3.6's second pair,
which used f16. The flow comparisons stand, because both arms of each pair share one KV type. The counts carry the
`q8_0` caveat (#387): under f16, gemma4:26b's `multi_3img_anchored` loops as well (item 8's GGUF leg, below).

The arms are `fold` and `fold2p` on this image, in production's environment, over the full ladder, interleaved.
llama-server drafts in neither arm: every server log reads `no implementations specified for speculative decoding`.
**On this host, the two arms differ in the flow and nothing else.**

**The loop probe.** Each arm was captured cold at 16384 with `thinkcap.py`, which uses the suite's own `gen()`.
`probe_readout.py`, verbatim:

```
== gemma4:31b-it-q4_K_M scene_single_pinned
  ladder fold2p: rung num_ctx=16384 done=stop eval=2359 json_valid=True think=3813 answer=1233 not_converged_at=None
  cap16k fold2p: done=stop eval=2359 think=3813 answer=1233 | 74 lines, 69 distinct, top x6: "*   Let's refine:"
  ladder fold  : rung num_ctx=16384 done=stop eval=2364 json_valid=True think=3813 answer=1278 not_converged_at=None
  cap16k fold  : done=stop eval=2364 think=3813 answer=1278 | 74 lines, 69 distinct, top x6: "*   Let's refine:"
  thinking: BYTE-IDENTICAL between arms (3813 chars)
== gemma4:31b-it-q4_K_M multi_3img_anchored
  ladder fold2p: rung num_ctx=16384 done=stop eval=3988 json_valid=True think=5858 answer=3505 not_converged_at=None
  cap16k fold2p: done=stop eval=4755 think=7889 answer=3478 | 142 lines, 132 distinct, top x3: '- Key objects:'
  ladder fold  : rung num_ctx=16384 done=stop eval=5236 json_valid=True think=8636 answer=3476 not_converged_at=None
  cap16k fold  : done=stop eval=4758 think=7889 answer=3486 | 142 lines, 132 distinct, top x3: '- Key objects:'
  thinking: BYTE-IDENTICAL between arms (7889 chars)
== gemma4:26b-a4b-it-q4_K_M scene_single_pinned
  ladder fold2p: rung num_ctx=16384 done=stop eval=5256 json_valid=True think=10145 answer=1252 not_converged_at=None
  cap16k fold2p: done=stop eval=5256 think=10145 answer=1252 | 217 lines, 210 distinct, top x3: '- Kind: rectangle'
  ladder fold  : rung num_ctx=16384 done=stop eval=5258 json_valid=True think=10145 answer=1271 not_converged_at=None
  cap16k fold  : done=stop eval=5258 think=10145 answer=1271 | 217 lines, 210 distinct, top x3: '- Kind: rectangle'
  thinking: BYTE-IDENTICAL between arms (10145 chars)
== gemma4:26b-a4b-it-q4_K_M multi_3img_anchored
  ladder fold2p: rung num_ctx=32768 done=stop eval=8335 json_valid=True think=15788 answer=3396 not_converged_at=None
  cap16k fold2p: done=stop eval=8335 think=15788 answer=3396 | 443 lines, 298 distinct, top x5: '"Payment due within 30 days. Quote reference INV-2026-0801 on all corr'
  ladder fold  : rung num_ctx=32768 done=stop eval=8331 json_valid=True think=15788 answer=3389 not_converged_at=None
  cap16k fold  : done=length eval=8192 think=15788 answer=3084 | 443 lines, 298 distinct, top x5: '"Payment due within 30 days. Quote reference INV-2026-0801 on all corr'
  thinking: BYTE-IDENTICAL between arms (15788 chars)
```

- **The two arms' thinking is byte-identical in all four cells, and nothing loops.**
- **The budget semantics change** appears on gemma4:26b `multi_3img_anchored` at 16384:
  - two-pass finishes at 8335 tokens, because pass two runs past `num_predict`;
  - the single pass stops at `length` 8192 with the same thinking;
  - the ladder finishes the single pass at 32768.

**gemma4:31b, full suite.**
- **The payload effect is nil.** Two-pass on b10969 against two-pass on b11081: 2 of 984 cells differ, both
  descriptive.
- **The flow moves 8 scored cells, all IoU noise, in both directions.** All 27 blocks end in `stop` with valid JSON
  in both flows.
- **Inside the suite, the thinking differs in 16 of 27 blocks.** Cold, it is byte-identical. An inference from the
  slot logs: the two-pass flow's second request lands on the other parallel slot. That changes the KV layout that
  later cells see, and the reduction order tips near-ties.

**gemma4:26b.**
- Both arms leave **one** case NOT CONVERGED at 131072, the same one: `bbox_contract_real_1img`. Every other case ends
  with `stop` and valid JSON, on the 16384 rung or the 32768 rung.
- It is a loop: 70 of 762 lines are distinct, and `*   ANCHOR: x1=72, y1=150, x2=216, y2=336.` repeats 98 times.
- Cold at 32768, it is **byte-identical in both flows and on b10969**. So it predates the fold, and production 0.34.3
  has it today. Its trigger is the prompt: with the image's size stated, gemma4:26b answers in 2,081 tokens with six
  correct pixel boxes (#387).
- **The flow moves no scored cell by more than 0.005.** Four IoU cells move by 0.001 to 0.002 toward two-pass. The
  thinking is identical in 23 of 27 blocks; the rest differ in the suite through the KV-layout effect described for
  31b.
- **The payload effect is nil.** 0 of 746 cells differ between two-pass on b10969 and two-pass on b11081, over the 23
  blocks that both finished on the 16384 rung.
- Metal's MLX count for 26b was 6 unfinished in the single pass against 1 in two-pass. Here it is 1 and 1: **where
  neither flow drafts, the flows loop equally.**

**qwen3.8 and nemotron3** have no sampling card, so they run at packaged sampling and are compared as rates, n = 2 runs
per arm. `rates2.py` (in the run directory) pools each arm's two runs, verbatim:

```
== qwen3.8:27b-q4_K_M
                                       fold2p             fold   (blocks 54 vs 54)
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
  answer_chars                        857.000          863.463
  bbox_mean_iou                         0.976            0.979
  eval_count                         1288.389         1275.815
  hits_anchor                           3.300            3.300
  hits_bestfit                          4.525            4.875
  hits_declared                         6.000            6.000
  iou_anchor                            0.527            0.528
  iou_at_implied_scale                  0.939            0.957
  iou_declared                          0.960            0.960
  labels_found                          6.000            6.000
  name_bbox_mean_iou                    0.387            0.772
  thinking_chars                     1906.833         1867.500
  done_reason                    {'stop': 54}     {'stop': 54}
  NOT CONVERGED                             -                -
== nemotron3:33b-q4_K_M
                                       fold2p             fold   (blocks 54 vs 54)
  anchor_beats_declared            7/40             4/40      
  anchor_present                  20/40            22/40      
  contract_followed               15/40            23/40      
  declaration_matches_boxes       15/40            23/40      
  declaration_valid               21/40            33/40      
  invoice_no                       2/2              2/2       
  json_valid                      54/54            54/54      
  q1_right                         4/4              4/4       
  q2_right                         4/4              4/4       
  q4_bbox_hit                      4/4              3/4       
  self_check                      14/20            15/22      
  serial_found                     5/6              5/6       
  think                           54/54            54/54      
  total_right                      2/2              2/2       
  answer_chars                       1172.833         1104.241
  bbox_mean_iou                         0.604            0.592
  eval_count                         5109.944         5168.778
  hits_anchor                           2.325            2.900
  hits_bestfit                          4.675            5.100
  hits_declared                         2.475            4.000
  iou_anchor                            0.280            0.317
  iou_at_implied_scale                    nan            0.049
  iou_declared                          0.300            0.441
  labels_found                          5.674            5.935
  name_bbox_mean_iou                    0.123            0.006
  thinking_chars                    11910.704        12275.019
  done_reason                    {'stop': 54}     {'stop': 54}
  NOT CONVERGED                             -                -
```

- **qwen3.8: every boolean rate is identical**, and all four runs finished on the 16384 rung. `name_bbox_mean_iou`
  moves from 0.387 to 0.772 toward the single pass, but it rests on `document_single` alone at n = 2, so it is noise
  until it repeats.
- **nemotron3: nothing is lost.** In both flows all 54 blocks end with `stop` and valid JSON; the ladder resolved every
  cell that capped at 16384 (5–7 per run). The single pass follows the bbox contract more often
  (`contract_followed` 23/40 against 15/40, `declaration_valid` 33/40 against 21/40). At n = 2 on a sampled model
  that is p ≈ 0.1 (Fisher, two-sided): no regression, possibly better, not established.

**qwen3.6** runs greedy on its card and is compared cell by cell. The pair ran twice: under `q8_0`, which was this
host's production KV cache from 2026-08-08 until it was found on 2026-09-26 (#386), and under f16, production's
setting now (#387). Derived from `done_reason` through `was_capped`, not generator output:

- `q8_0`: both flows finish 25 of 27 cases. `bbox_contract_real_1img` and `bbox_contract_adv_real` never finish at
  131072, in either flow.
- f16: both flows finish 26 of 27. Only `bbox_contract_real_1img` never finishes, in either flow. `adv_real` finishes
  in 16,272 tokens (single pass) and 10,312 (two-pass).

The flow's scored effect, from #387's `cmp_scored.py`, verbatim: quality fields only, over the cases that finished in
both arms, A = `fold` (single pass) and B = `fold2p` (two-pass).

```
== q8_0
bbox_contract_box2d_1img: iou_anchor 0.633->0.892 B+; iou_declared 0.633->0.892 B+; self_check False->True
bbox_contract_positional_1img: iou_anchor 0.635->0.616 A+; iou_declared 0.635->0.616 A+
bboxm_free_noanc_pos: iou_declared 0.836->0.957 B+
bboxm_pin_anc_pos: iou_anchor 0.57->0.968 B+; iou_declared 0.57->0.968 B+
multi_3img: q4_bbox_space 'norm1000/xyxy'->'pixel/xyxy'
quality moves: 5 favour B, 2 favour A, 2 label changes (A = scores_r0344p_fold_1_qwen3_6_35b-a3b-q4_k_m_thinkon.json, B = scores_r0344p_fold2p_1_qwen3_6_35b-a3b-q4_k_m_thinkon.json)
== f16
bbox_contract_adv_norm1: iou_anchor 0.931->0.968 B+; iou_declared 0.931->0.968 B+
bbox_contract_adv_real: hits_anchor 0->2 B+; hits_declared 0->2 B+; iou_anchor 0.155->0.305 B+; iou_declared 0.155->0.305 B+
bbox_contract_anchored: iou_anchor 0.899->0.6 A+; iou_declared 0.899->0.6 A+; self_check True->False
bbox_contract_box2d_1img: iou_anchor 0.647->0.967 B+; iou_declared 0.647->0.967 B+
bbox_contract_perobject: iou_declared 0.652->0.964 B+
bbox_contract_reasoning: declared_ref [1000, 562]->[1000, 1000]; declared_type 'real'->'norm1000'; hits_bestfit 1->6 B+; iou_declared 0.923->0.676 A+
bboxm_free_anc_pos: iou_anchor 0.947->0.97 B+; iou_declared 0.947->0.97 B+
quality moves: 12 favour B, 3 favour A, 3 label changes (A = scores_r0344pf16_fold_1_qwen3_6_35b-a3b-q4_k_m_thinkon.json, B = scores_r0344pf16_fold2p_1_qwen3_6_35b-a3b-q4_k_m_thinkon.json)
```

- **No loop regression from the single pass**, under either KV cache.
- **The quality moves lean toward two-pass:** 3 of 4 cases under `q8_0`, and 5 of 7 under f16 with one mixed. Only
  `bbox_contract_box2d_1img` moves the same way under both (0.633 and 0.647 against 0.892 and 0.967). The rest change
  with the KV type, which by itself moves the quality of 20 of 25 cases (#387). So it is a lean, not a verdict.
- **The loops come from the prompt.** `bbox_contract_real_1img` asks for "the size YOU used" after an internal resize
  the model cannot see, and it loops under every KV type and attention path measured (#387). Cold, `adv_real`
  finishes under both KV types; its `q8_0` loop in the suite came from the run's state.

**Open item 8, the GGUF leg: `multi_3img_anchored`** (2026-09-27, at the maintainer's word). This was gemma4:26b-a4b
q4_K_M on the fold image, with production's settings: f16 KV and flash attention on, single pass. Each case was
captured cold by `promptcap.py` (#387, `b13f2c35d`), greedy, at 32768 (24,576 tokens). The `orig` prompts'
fingerprints equal the suite's `prompt_sha`. `item8_read.py` (in the run directory) reads each capture with #387's
`kvloop_read.py` and scores it with `score_multi`, verbatim:

```
multi_3img orig: done=stop tokens=3616 thinking=5618 answer=3086 chars
  thinking: 90/102 lines distinct, second half 42/51, most repeated x3: '- **Key Objects:**'
  onset: no loop found
  score_multi: json_valid=True q1_right=True q2_right=True q4_bbox_hit=True q4_bbox_space=norm1000/xyxy chart_values_found=5
  __IMAGE__ anchor: None
multi_3img_anchored orig: done=length tokens=24576 thinking=61234 answer=0 chars
  thinking: 239/1717 lines distinct, second half 11/859, most repeated x288: 'The text "DYNAMO" is at y ~ 530.'
  onset: loop from line 280 of 1,717, about token 4,148
  score_multi: json_valid=False q1_right=False q2_right=False q4_bbox_hit=False q4_bbox_space=None chart_values_found=0
  __IMAGE__ anchor: unparsed (AttributeError)
multi_3img_anchored size: done=stop tokens=5875 thinking=9557 answer=3510 chars
  thinking: 177/239 lines distinct, second half 93/120, most repeated x4: '"ANCHOR"'
  onset: no loop found
  score_multi: json_valid=True q1_right=True q2_right=True q4_bbox_hit=True q4_bbox_space=norm1000/xyxy chart_values_found=5
  __IMAGE__ anchor: [0, 0, 1920, 1080]
multi_3img_anchored commit: done=stop tokens=5633 thinking=10023 answer=3431 chars
  thinking: 174/217 lines distinct, second half 107/109, most repeated x3: '- Key objects:'
  onset: no loop found
  score_multi: json_valid=True q1_right=True q2_right=True q4_bbox_hit=True q4_bbox_space=norm1000/xyxy chart_values_found=5
  __IMAGE__ anchor: [0, 0, 1000, 1000]
```

- **The trap sentence alone makes the case loop, and replacing it ends the loop.**
  - The control, `multi_3img`, finishes in 3,616 tokens.
  - `multi_3img_anchored` loops from about token 4,148 and never answers. It re-derives image 1's height: "The text
    "DYNAMO" is at y ~ 530.", "The orange circle is at y ~ 554 to 799." and "This means the image height is at least
    800." repeat 288 times each, and the second half has 11 distinct lines.
  - With only that sentence replaced, it finishes in 5,875 tokens (`size`) or 5,633 (`commit`), with every question
    right.
- **Under `q8_0` the same case finished.** The protocol's cells end at 8,331 and 8,335 tokens, and their 15,788
  characters of thinking equal the cold probe's above. On this host the KV type decides the case, and production's
  f16 is the setting that loops. That matches #387: where a loop starts moves with the numerical path, and the prompt
  sets the trap.
- **The anchor tells which frame the model used.** With the size stated, gemma4 declares `[0, 0, 1920, 1080]` but
  answers in 0–1000, which is the "anchor lies" case in the suite's README; the scorer still finds the box, in
  norm-1000. With the commit instruction it declares `[0, 0, 1000, 1000]`, the frame it answers in.

Throughput is not a finding on this host, because the GPU was shared.

## Found on `main`, not caused by this fold

Sweeping the merged tree for the old representation found **three sites ADR 0039 missed**, all from upstream's MLX
bump of 2026-09-14, six days before ADR 0039 landed:

| site | what it does | under ADR 0039 |
|---|---|---|
| `mlxrunner/model/laguna/laguna.go:731` | divides a `QuantizedLinear.GlobalScale` by 2688 | that field holds `m`, so dense laguna nvfp4 output is ×1/2688 |
| `mlxrunner/model/nemotron_h/nemotron_h.go:447` | divides the loader's scale by 2688, "to undo MLX's representation" | `ReadGlobalScale` returns `LoadGlobalScale(found)` = `m`; experts are ×1/2688 |
| `mlxrunner/model/qwen3_5/gdn_projections.go:195` | fills a missing scale with 2688 "in MLX's representation" beside present ones in `m` | a mixed bank; reachable only when one of a hi/lo pair lacks a scale |

**None is on a model production serves**, and production's own campaign agrees: qwen3.8:27b-nvfp4 runs through
`gdn_projections.go` and scored 0.99 IoU, so its pairs always carry both scales. `TestApplyExpertWeightGlobalScale`
passes because its fixture feeds the MLX form the loader no longer produces. These get their own change with a
representation-sensitive test each, so the fold's attribution stays clean.

## Open items

1. **Gate 6 on CUDA is done** (2026-09-27 17:40). The fixed-history MLX think-on variant finished last, after the
   drafting probe (item 7) and #387's KV-precision × flash-attention test. All three are under gate 6.
2. **`ce8caa6e6`: the device half is carried as compat patch 908**, on the maintainer's word (2026-09-26). That half,
   the tiling, is five extra never-ending think-on loops on gemma4:26b (6 of 27 against 1, and gfx1151's 1), the
   think-off movement of gemma4:31b, 26b and e4b, and the GGUF q4 OCRBench item; on gemma4:31b's think-on it changes
   numerics only. The host half, the
   decode selection, stays upstream's: it moves qwen3.6's think-off cells in both directions (11 better, 9 worse) and
   causes no loop. **The image is rebuilt with it** (`maxusai/ollama:sync-0.34.4-908`, `90bb7ffc0be6`,
   `0.34.3-dynres-22-g5584539`). Against the tested candidate, three of its 2,697 payload files differ: `bin/ollama`
   and the two `libggml-cuda.so`. Its sm_120a PTX is identical to the device-half library every measurement used, in
   all 6,240 kernels, once CUB's and Thrust's ABI tags are normalised (they encode the compiled-architecture list).
   **Gates 5 and 6 pass on it** (above): preflight PASS=21 SKIP=8, and GGUF think-off equals production on gemma4:31b,
   26b and e4b and on nemotron3, while e2b and the qwen models keep the host half's movement. The ROCm host's check is done: 908 changes no
   gfx1151 kernel, and preflight passes on the rebuilt image (gates 4 and 5 on gfx1151).
3. **The think+format default on MLX.** Metal's single pass with drafting on (`OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=1`) runs
   next on the full 26b and 31b suites, and separates the flow from the drafting. An ADR superseding ADR 0004 follows
   the data, and records the budget semantics listed above.
4. **`TestMulGatherQMMGlobalScale`** is gated to Metal upstream and passes on CUDA with the gate widened (128.8 s);
   the widening is its own change. So are the three ADR 0039 misses above.
5. **Gate 6 think-on on gfx1151**: qwen3.8, nemotron3 and qwen3.6 under the aligned protocol, into 2026-09-26.
   **Gates 4 and 6 on Metal**, on the host that owns them.
6. **The CUDA deploy sets `OLLAMA_KV_CACHE_TYPE=f16` explicitly** (the maintainer's decision, 2026-09-26, after #386
   and #387). Production does not set the variable today; it runs the default, and its log shows f16 in all 12 of its
   KV cache allocations, so it is not recreated for this alone. The v0.34.4 deploy mirrors production's container by
   `docker inspect`, as every CUDA deploy has, and adds the variable, and item 7's `OLLAMA_FORMAT_TWO_PASS=1`. If the
   live container carries a different value, it refuses rather than choose. The deploy itself waits on the
   maintainer's word.
7. **The deploy's think+format configuration on MLX: two-pass** (the maintainer's decision, 2026-09-27, after the
   drafting probe under gate 6). The v0.34.4 deploy sets `OLLAMA_FORMAT_TWO_PASS=1` beside item 6's f16. That keeps
   today's speed, today's memory behaviour and ADR 0004's flow. Mirroring production alone would have run the single
   pass with `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`, which is arm F0, and thinks 1.5–1.7× slower than production's
   two-pass today. `deploy-v0344.sh` adds the switch the way it adds f16:
   - It refuses if the live container carries another value.
   - It rolls back unless the new container's environment carries the switch and the server's startup config reads
     `OLLAMA_FORMAT_TWO_PASS:true`.

   It was chosen over three other options:
   - **The knob at 1**: F1's speed. But drafting under a grammar brings back the 0.34.1 record's drafting retention
     (an image, a stop and speculation) for think-off structured image requests on the qwen3.5 family, which are the
     requests the knob was deployed for.
   - **Draft while the grammar is still in its free-thinking element, and stop drafting at the closing**: the
     drafting today's pass one does, in one pass. This is a code change in `mlxrunner`, and the size ladder
     (`leak-repro5.sh`) gates it.
   - **A per-family knob.** In the 0.34.1 fold's `held` series (gate 5b's runner log), gemma4:31b drafted under a
     grammar on all 28 of its image requests (15,456 draft tokens) and 12b on 24 of them, and both held flat. qwen3.6
     grew 0.60 GiB per request in the same run, which the retention later accounted for. So gemma4 could draft under a
     grammar while the qwen3.5 family keeps the knob. gemma4:26b drafted only 96 tokens in that run, so it needs its
     own ladder.
8. **Whether the trap sentence alone makes `multi_3img_anchored` loop on CUDA.** `promptcap.py` covers this case
   since #387's `b13f2c35d`, stating image 1's size (1920×1080). The check is three captures on gemma4:26b, on MLX
   and on GGUF: `orig`, `size` and `commit` at 32768. `multi_3img`'s `orig` capture is the control.
   - **The GGUF leg ran on gfx1151 (2026-09-27): yes.** With f16 KV, `orig` loops from about token 4,148. `size` and
     `commit` finish in 5,875 and 5,633 tokens with every answer right. The control finishes in 3,616.
     See [the gfx1151 section](#gate-6-think-on-under-the-aligned-protocol).
   - **The CUDA legs ran (2026-09-27).** On GGUF the answer depends on the image. On the fold image the sentence
     alone makes the case loop, as on gfx1151. On the image that ships, all four captures finish with every question
     right, because the loop also needs `ce8caa6e6`'s tiling, which 908 reverts (item 2). On MLX, five cold draws of
     each prompt, `orig`, `size` and `commit` each finished 1 of 5 and the control 3 of 5. So the sentence does not
     set MLX's loop rate; stating the size changes only how the case loops. See
     [Open item 8 on CUDA](#open-item-8-on-cuda-the-trap-sentence).
