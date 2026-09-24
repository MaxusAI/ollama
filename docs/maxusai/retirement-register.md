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
| **think + format at the routes layer**: the marker-stop flow (pass one stops at the think-close tag and continues textually), pass-one metrics reconstruction, the second pass pinned to pass one's truncation window | ADR 0002, 0004, 0010 | upstream's own transition flow (`structuredOutputsState`, in 0.34.1 with pass-one metrics folded into the final response) covers marker models and every runner reports pass-one metrics itself | the pass-one suite in `server/routes_generate_test.go`; preflight `think_format`; the `bcreasoning` cells — our flow against upstream's on the same models | upstream has the transition flow and the metrics merge; ours remains a superset |
| **gemma4 vision on MLX through our path**, upstream's tower excluded | ADR 0021, D1-A; the D1-B spike | upstream's tower gains a per-request image budget, or the budget seam is re-grafted onto it (D1-B) | vision goldens 12b/26b/31b; the MLX think-off campaign | 1,523/−2,271 lines against upstream; D1-B not started |
| **fp32 cuBLAS accumulation for the shared Qwen-VL clip graph** (arch `qwen2vl` and `qwen25vl` — one family, two converter spellings; widened 2026-09-19) | #214, ollama#18070 | #18070 or a per-op `PREC_F32` fix lands upstream | `poison_probe`; the fp16 canary | #18070 open |
| **`903` MMQ ids-path padding** | llama.cpp#27044 | fixed upstream | the MoE + MMQ functional gate | #27044 open |
| **`801` clip node-stats meter** | #214 diagnostic | with the f32 gate above; it exists to localise that overflow | the fp16 canary under `OLLAMA_CLIP_NODE_STATS` | keep while the gate is carried |
| **`004` gemma4 budget fill** (`image_budget_fill`, `PAD_NONE`) | ADR 0008 | upstream fills to the ladder and stops letterbox-padding. Its default *limits* converged on ours (70/1120) in b10864 | `pinned_image_token_budget`, `token_ladder`, bbox conformance | limits converged; the fill is still ours |
| **`002` nemotron native-aspect dynamic resolution** | ADR 0001 | upstream leaves the fixed 512×512 canvas | `token_ladder`; nemotron bbox cells | fixed canvas in llama.cpp through b10969, where 002 applies clean; upstream's **MLX** nemotron_h is native-aspect since v0.34.3 (#17714), 1024…13312 patches, the same bounds |
| **ADR 0014's fork-local half**: `growOpeningChunk` and the all-or-nothing match for bidirectional expansions | ADR 0014 | upstream's non-causal boundary work (0.34.1) covers bidirectional expansions | `media_test.go`, the prefix-cache media tests, gemma4 multi-image cells | partly converged: `extendChunk` is upstream's |
| **MLX admission** (KV priced at the rung, per-arch headroom, refuse/clamp) and the **memory and cache limits** | ADR 0034 | upstream prices KV at admission and exposes the limits | `client_admission_test.go`, `client_budget_test.go`; the OOM ladders | upstream bounds by free memory only; it generates the limit symbols but does not call them |
| **stop sequences on MLX** | `stopper.go` | upstream's MLX runner honours `stop` | `stopper_test.go` | none upstream |
| **per-model `kv_cache_type`** | ADR 0005 | upstream adds a per-model option | `llm/kv_cache_type_test.go` | one global env upstream |
| **scheduler fixes**: `evictAbandoned`, head-of-line, leaf `logMu` | `server/sched.go` | fixed upstream | `sched_test.go`, `sched_headofline_test.go` | fork-only |
| **capability rule: gemma4 safetensors without audio** | `server/images.go`, ADR 0021 | the gemma4-on-MLX row above retires — upstream's tower brings its audio path | `images_test.go` | fork-only. The nemotron text-only half is retired (below) |
| **gemma4 image chunk vs. batch**: the launcher's `-b N -ub N` keeps llama.cpp #28954's abort away, and since ADR 0036 the scheduler sizes the batch to the image ceiling so the non-causal chunk decodes in one piece | task doc 0.34.1, 2026-09-18 | upstream fits an image chunk to one ubatch (llama.cpp #28954's fix) or sizes ubatch to the image cap | the top-rung bbox cells at `num_batch` 1024 vs 2048; preflight `token_ladder` | #28954 open; **fixed in the fork 2026-09-18** (ADR 0036, #320): a gemma4 vision runner starts from the batch rung holding its image ceiling, measured first (no contract cost, 1.5–1.9× prefill, 31b 9 px tier 4 → 3) |
| **the drafting knob as memory mitigation** (`OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`, deployed 2026-09-18): the MLX runner retains 0.1–0.3 GiB per image request that ends by stop under speculation (component A, grammar or not); the knob removes it for grammar requests only | task doc 0.34.1 "drafting retention"; `upstream-mlx-speculation-image-stop-retention.md` (parked) | upstream finds and frees the holder on MLX's side | the size ladder (`leak-repro5.sh`): `held − trie` flat with drafting on | reproduced, exclusions measured, no fix known |
| **transparent images composited over white** (gemma4) | ADR 0015 | upstream's gemma4 path composites; its qwen3.5 path drops alpha by RGB conversion instead, per that model's reference | gemma4 vision goldens | not converged |

## Fork tooling with no upstream counterpart

| item | retires when | status |
|---|---|---|
| `Dockerfile.gemma4budget` + `docs/maxusai/gemma4-budget-image.md` | the overlay cannot carry payload patches, and every fold has built full images since 0.32 | stale; Glenn's call |
| `Dockerfile.applearm` + `docs/maxusai/spec/apple-silicon-build.md` | the Metal half is built another way, or dropped | held with the Metal half |
| `Dockerfile.rocm` + `scripts/build_rocm.sh` ([ADR 0042](adr/0042-rocm-images-build-on-ubuntu-rocm-images.md)) — every ROCm image stage on `rocm/dev-ubuntu-24.04` (7.2.4-complete / 10.0.0-full), never AlmaLinux | upstream builds ROCm on an image AMD still publishes and the two recipes are shown equivalent, or Glenn lifts the rule | **until further notice** (2026-09-24); each fold diffs upstream's ROCm stages against it |
| `rocm_v10_0` presets beside `rocm_v7_2` (`llama/server/CMakePresets.json`) and ROCm 10's split runtime libraries in `llama/server/CMakeLists.txt`'s bundling regexes ([ADR 0042](adr/0042-rocm-images-build-on-ubuntu-rocm-images.md), from #359) | upstream ships a ROCm 10 preset and bundles those libraries itself | carried; the regexes match nothing on ROCm ≤ 7.2 |

## Already retired

- **`x/structured`** (ADR 0009/0013), deleted 2026-09-17 in the v0.34.1 fold on Glenn's word, after the parity gate
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
