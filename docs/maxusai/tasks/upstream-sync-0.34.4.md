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
| 5, preflight | **PASS on CUDA**: run 2 PASS=21 SKIP=8, with the pins moved in `c79e50d98` after run 1 (FAIL=2 on the two pins, by design). **gfx1151: PASS=20 SKIP=12** with the new `rocm7-0-34-4-dynres` (#378), and the same on the 908 image |
| 6, campaigns | **in progress on CUDA.** GGUF think-off: 210 of 6,909 cells move, in the head-dimension-256 models only, and reverting `ce8caa6e6` restores production on both probes: its device half on gemma4:31b, its host half on qwen3.6. MLX think-off: no consistent difference, inside or across MLX-CUDA's run-to-run spread. OCRBench: production's scores, but for one reproducible item on GGUF q4, which is `ce8caa6e6`'s device half. Think-on: `ce8caa6e6`'s device half leaves 6 of 27 gemma4:26b cases in loops that never end, against 1 without it; 31b is unaffected. See [Gates 4–6 on CUDA](#gates-46-on-cuda-2026-09-25). **gfx1151:** think-off and OCRBench equal production in every scored cell. Think-on under the aligned protocol is in progress, with no single-pass regression so far. See [Gates 4–6 on gfx1151](#gates-46-on-gfx1151-2026-09-25) |

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
124 lines in one CUDA source file). It removes the loop and restores gemma4's think-off cells, but not qwen3.6's,
which move with the host half (above); the host half stays upstream's.

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

### Gate 6: think on, under the aligned protocol (in progress)

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
- The two-pass arm leaves **one** case NOT CONVERGED at 131072: `bbox_contract_real_1img`.
- It is a loop: 70 of 762 lines are distinct, and `*   ANCHOR: x1=72, y1=150, x2=216, y2=336.` repeats 98 times.
- Cold at 32768, it is **byte-identical in both flows and on b10969**. So it predates the fold, and production 0.34.3
  has it today.
- The single-pass arm is running.

**Queued, into 2026-09-26:** qwen3.8 (n = 2 per arm), nemotron3 (n = 2), then qwen3.6. qwen3.6's think runaway under
q8_0 KV gives it the longest ladder.

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

1. **Gate 6 on CUDA, queued, in this order:** MLX think-on under the agreed protocol (running), gates 5 and 6 on the
   908 image (preflight, then GGUF think-off on all eight models), the drafting probe, the KV-precision ×
   flash-attention loop test that #387 asks for (gemma4:26b, with the tiling and without it), and a fixed-history MLX
   think-on variant with every case as the first request after a cold restart, for Metal's finding that request
   history moves MLX's loops.
2. **`ce8caa6e6`: the device half is carried as compat patch 908**, on the maintainer's word (2026-09-26). That half,
   the tiling, is five extra never-ending think-on loops on gemma4:26b (6 of 27 against 1, and gfx1151's 1), gemma4's
   think-off movement and the GGUF q4 OCRBench item; on gemma4:31b it changes numerics only. The host half, the
   decode selection, stays upstream's: it moves qwen3.6's think-off cells in both directions (11 better, 9 worse) and
   causes no loop. **The image is rebuilt with it** (`maxusai/ollama:sync-0.34.4-908`, `90bb7ffc0be6`,
   `0.34.3-dynres-22-g5584539`). Against the tested candidate, three of its 2,697 payload files differ: `bin/ollama`
   and the two `libggml-cuda.so`. Its sm_120a PTX is identical to the device-half library every measurement used, in
   all 6,240 kernels, once CUB's and Thrust's ABI tags are normalised (they encode the compiled-architecture list).
   Still to do for 908 on CUDA: preflight on that image, and the GGUF think-off cells re-run on it (queued, item 1). The ROCm host's check is done: 908 changes no
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
   `docker inspect`, as every CUDA deploy has, and adds the variable. If the live container carries a different value,
   it refuses rather than choose. The deploy itself waits on the maintainer's word.
