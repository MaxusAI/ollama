# Retirement register: what the fork carries, and what retires it

The README's promise is that every fork delta is "offered upstream where it is upstream's to take, and deleted from
here once it lands there". This is the list that makes that checkable at each fold. One row per carried item: what
retires it, and the test that must pass before it is deleted — deletion is a change like any other and gets a gate.
Reviewed at every fold; last against upstream v0.34.3 / llama.cpp b10969 (2026-09-24).

## Retire now, once tested

_Nothing pending. `x/structured` moved to "Already retired" below on 2026-09-17._

## Carried until upstream converges

| item | record | retires when | test that gates deletion | status 2026-09-24 |
|---|---|---|---|---|
| **think + format, two-pass** (ADR 0004's routes-layer flow, behind `OLLAMA_FORMAT_TWO_PASS=1` since v0.34.4): the marker-stop flow (pass one stops at the think-close tag and continues textually), pass-one metrics reconstruction, the second pass pinned to pass one's truncation window, and the lower-layer hooks the v0.34.4 fold restored | ADR 0002, 0004, 0010, [0045](adr/0045-think-format-single-pass-by-default-two-pass-in-production.md) | a single pass drafts its thinking on MLX without the qwen3.5-family retention, so production can drop the switch (ADR 0045, proposed) | `leak-repro5.sh` flat on the qwen3.5 family; the drafting probe with the single pass matching P0; the MLX think-on protocol with the single pass matching two-pass's loop counts, on Metal; then the two-pass route tests go with the flow | 2026-09-28: upstream's flow is a single pass since v0.34.4, and ours is production's flow (ADR 0045) |
| **gemma4 vision on MLX through our path**, upstream's tower excluded | ADR 0021, D1-A; the D1-B spike | upstream's tower gains a per-request image budget, or the budget seam is re-grafted onto it (D1-B) | vision goldens 12b/26b/31b; the MLX think-off campaign | 1,523/−2,271 lines against upstream; D1-B not started |
| **fp32 cuBLAS accumulation for the shared Qwen-VL clip graph** (arch `qwen2vl` and `qwen25vl` — one family, two converter spellings; widened 2026-09-19) | #214, ollama#18070 | #18070 or a per-op `PREC_F32` fix lands upstream | `poison_probe`; the fp16 canary | #18070 open |
| **`903` MMQ ids-path padding** | llama.cpp#27044, closed 2026-10-04 for #29941; [submission material](upstream-mmq-submission-material.md), "Outcome" | upstream pads for every tile it launches, by the flattened row count or by the launched `J`, so that no shape is short | `tasks/mmq-ids-padding-gpu.sh` on the new pin: its `master` build, which is the pin's own line, at 0 memcheck errors with the exact allocation, the 65- and 100-token cases included; `tasks/mmq-ids-padding-test.cu` extended to that line and short in no shape; then the MoE + MMQ functional gate | 2026-10-04: upstream merged #29941 (`dd266785c`), which pads by `ne12`. That is enough from 128 tokens up, where every reported crash was, and short by up to 63 blocks below 128 (268,440 of 2,439,360 shapes; memcheck errors at 65 and 100 tokens on sm_120). Kept on the maintainer's word (2026-10-04). Re-cut at the first pin that contains `dd266785c`: its context line reads `ne12`, so 903 becomes `ne12` → `ne12*n_expert_used` |
| **`908` gemma4 FA tiling revert**: the device half of llama.cpp `ce8caa6e6` (MMA configs and tile sizes for head dims 256/512), back to b10969's | #375; the v0.34.4 fold record, "Gates 4–6 on CUDA" | a fold's CUDA GGUF loop-rate run on the pin's own tiling matches 908's. No upstream fix is due: `ce8caa6e6` loses no precision in `test-backend-ops` (2026-09-28) | the CUDA GGUF loop-rate run: gemma4:26b-a4b full think-on suite on the full ladder, the pin as shipped against the pin with 908, NOT CONVERGED counts (1 of 27 with 908, 6 without, at b11081); then the think-off cells of gemma4:31b, 26b and e4b against production (e2b moves with both halves) | carried from the v0.34.4 fold (2026-09-26) on the maintainer's word. Not an upstream bug: on 2026-09-28 the precision was unchanged, and on the public Q4_0 the loop runs the other way. Nothing is filed |
| **`801` clip node-stats meter** | #214 diagnostic | with the f32 gate above; it exists to localise that overflow | the fp16 canary under `OLLAMA_CLIP_NODE_STATS` | keep while the gate is carried |
| **`004` gemma4 budget fill** (`image_budget_fill`, `PAD_NONE`) | ADR 0008 | upstream fills to the ladder and stops letterbox-padding. Its default *limits* converged on ours (70/1120) in b10864 | `pinned_image_token_budget`, `token_ladder`, bbox conformance | limits converged; the fill is still ours |
| **`002` nemotron native-aspect dynamic resolution** | ADR 0001 | upstream leaves the fixed 512×512 canvas | `token_ladder`; nemotron bbox cells | fixed canvas in llama.cpp through b10969, where 002 applies clean; upstream's **MLX** nemotron_h is native-aspect since v0.34.3 (#17714), 1024…13312 patches, the same bounds |
| **ADR 0014's fork-local half**: `growOpeningChunk` and the all-or-nothing match for bidirectional expansions | ADR 0014 | upstream's non-causal boundary work (0.34.1) covers bidirectional expansions | `media_test.go`, the prefix-cache media tests, gemma4 multi-image cells | partly converged: `extendChunk` is upstream's |
| **MLX admission** (KV priced at the rung, per-arch headroom, refuse/clamp) and the **memory and cache limits** | ADR 0034 | upstream prices KV at admission and exposes the limits | `client_admission_test.go`, `client_budget_test.go`; the OOM ladders | upstream bounds by free memory only; it generates the limit symbols but does not call them |
| **stop sequences on MLX** | `stopper.go` | upstream's MLX runner honours `stop` | `stopper_test.go` | none upstream |
| **per-model `kv_cache_type`** | ADR 0005 | upstream adds a per-model option | `llm/kv_cache_type_test.go` | one global env upstream |
| **llama-server's prompt-cache size**: `prompt_cache_ram`, per model or per request, and the vision harness's default of `0` (`vision-suite/client.py`) | ADR 0047 | upstream passes `--cache-ram`, or adds a per-model option for it | `TestAppendPromptCacheArgs`, `TestSchedNeedsReloadPromptCacheRAM`; the harness's `test_client.py::TestPromptCacheRAM` | upstream passes no `--cache-ram` (2026-10-03) |
| **scheduler fixes**: `evictAbandoned`, head-of-line, leaf `logMu` | `server/sched.go` | fixed upstream | `sched_test.go`, `sched_headofline_test.go` | fork-only |
| **capability rule: gemma4 safetensors without audio** | `server/images.go`, ADR 0021 | the gemma4-on-MLX row above retires — upstream's tower brings its audio path | `images_test.go` | fork-only. The nemotron text-only half is retired (below) |
| **gemma4 image chunk vs. batch**: the launcher's `-b N -ub N` keeps llama.cpp #28954's abort away, and since ADR 0036 the scheduler sizes the batch to the image ceiling so the non-causal chunk decodes in one piece | task doc 0.34.1, 2026-09-18 | upstream fits an image chunk to one ubatch (llama.cpp #28954's fix) or sizes ubatch to the image cap | the top-rung bbox cells at `num_batch` 1024 vs 2048; preflight `token_ladder` | #28954 open; **fixed in the fork 2026-09-18** (ADR 0036, #320): a gemma4 vision runner starts from the batch rung holding its image ceiling, measured first (no contract cost, 1.5–1.9× prefill, 31b 9 px tier 4 → 3) |
| **the drafting knob as memory mitigation** (`OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`, deployed 2026-09-18): the MLX runner retains 0.1–0.3 GiB per image request that ends by stop under speculation (component A, grammar or not); the knob removes it for grammar requests only | task doc 0.34.1 "drafting retention"; `upstream-mlx-speculation-image-stop-retention.md` (parked) | upstream finds and frees the holder on MLX's side | the size ladder (`leak-repro5.sh`): `held − trie` flat with drafting on | reproduced, exclusions measured, no fix known |
| **transparent images composited over white** (gemma4) | ADR 0015 | upstream's gemma4 path composites; its qwen3.5 path drops alpha by RGB conversion instead, per that model's reference | gemma4 vision goldens | not converged |

## Fork tooling with no upstream counterpart

| item | retires when | status |
|---|---|---|
| `Dockerfile.gemma4budget` + `docs/maxusai/gemma4-budget-image.md` | the overlay cannot carry payload patches, and every fold has built full images since 0.32 | stale; the maintainer's call |
| `Dockerfile.applearm` + `docs/maxusai/spec/apple-silicon-build.md` | the Metal half is built another way, or dropped | held with the Metal half |
| `Dockerfile.rocm` + `scripts/build_rocm.sh` ([ADR 0042](adr/0042-rocm-images-build-on-ubuntu-rocm-images.md)) — every ROCm image stage on `rocm/dev-ubuntu-24.04` (7.2.4-complete / 10.0.0-full), never AlmaLinux | upstream builds ROCm on an image AMD still publishes and the two recipes are shown equivalent, or the maintainer lifts the rule | **until further notice** (2026-09-24); each fold diffs upstream's ROCm stages against it |
| `rocm_v10_0` presets beside `rocm_v7_2` (`llama/server/CMakePresets.json`) and ROCm 10's split runtime libraries in `llama/server/CMakeLists.txt`'s bundling regexes ([ADR 0042](adr/0042-rocm-images-build-on-ubuntu-rocm-images.md), from #359) | upstream ships a ROCm 10 preset and bundles those libraries itself | carried; the regexes match nothing on ROCm ≤ 7.2 |

## Already retired

- **`x/structured`** (ADR 0009/0013), deleted 2026-09-17 in the v0.34.1 fold on the maintainer's word, after the parity gate
  found 0 regressions against xgrammar v0.2.5 in 108 verdicts. Its one finding — `allOf` with several branches is
  permissive on xgrammar — is recorded in ADR 0033's amendment and pinned by
  `mlxrunner/xgrammar/engine_behaviour_test.go`, with the ADR 0013 bound kept there as a budget test.
- ADR 0017's mechanism (`mlx.ClaimOSThread`) — upstream's `mlxthread.Start` carries the guarantee since the 0.33.3 fold.
- ADR 0007 (gemma4 default budget 560) — superseded by ADR 0008.
- The integrated-GPU admission bound — upstream's, absorbed into `admit()` in the 0.34.1 fold.
- The nemotron text-only capability rule. Our copy lived in `server/model_list_cache.go` and went with that file in
  the 0.34.1 fold, leaving upstream's own `suppressVisionCapability` (#17060, since v0.32.9) in `images.go`; v0.34.3
  deletes that too, because nemotron_h serves vision on MLX (#17714). Nothing of ours remained to delete; the fold's
  `images_test.go` carries upstream's inverted expectation.
- gemma4's default image-token limits — upstream adopted 70/1120 in b10864; our flags still pass them, per request.
- **compat patch 906** (revert of "restore `prop.integrated` on HIP builds") — retired in the v0.34.2 fold. The patch
  carried upstream llama.cpp's own revert `d4389a4dd92`, which our b10864 payload missed by 78 minutes; b10969 ships
  it, so the source already reads `info.devices[id].integrated = false` and the patch no longer applies. Dropping it
  on that evidence is what the patch's own header asked for: "Drop this patch when the payload advances past
  d4389a4dd92." The defect it guarded — an MMQ tile-barrier race on gfx1151 producing wrong output past `n_ubatch`,
  measured here as gemma4:31b `name_bbox_mean_iou` 0.728 → 0.000 at the 1120 budget — stays fixed by upstream's code.
