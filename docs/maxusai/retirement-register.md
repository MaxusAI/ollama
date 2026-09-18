# Retirement register: what the fork carries, and what retires it

The README's promise is that every fork delta is "offered upstream where it is upstream's to take, and deleted from
here once it lands there". This is the list that makes that checkable at each fold. One row per carried item: what
retires it, and the test that must pass before it is deleted — deletion is a change like any other and gets a gate.
Reviewed at every fold; last against upstream v0.34.1 / llama.cpp b10864 (2026-09-17).

## Retire now, once tested

_Nothing pending. `x/structured` moved to "Already retired" below on 2026-09-17._

## Carried until upstream converges

| item | record | retires when | test that gates deletion | status 2026-09-17 |
|---|---|---|---|---|
| **think + format at the routes layer**: the marker-stop flow (pass one stops at the think-close tag and continues textually), pass-one metrics reconstruction, the second pass pinned to pass one's truncation window | ADR 0002, 0004, 0010 | upstream's own transition flow (`structuredOutputsState`, in 0.34.1 with pass-one metrics folded into the final response) covers marker models and every runner reports pass-one metrics itself | the pass-one suite in `server/routes_generate_test.go`; preflight `think_format`; the `bcreasoning` cells — our flow against upstream's on the same models | upstream has the transition flow and the metrics merge; ours remains a superset |
| **gemma4 vision on MLX through our path**, upstream's tower excluded | ADR 0021, D1-A; the D1-B spike | upstream's tower gains a per-request image budget, or the budget seam is re-grafted onto it (D1-B) | vision goldens 12b/26b/31b; the MLX think-off campaign | 1,523/−2,271 lines against upstream; D1-B not started |
| **fp32 cuBLAS accumulation for `qwen25vl`** | #214, ollama#18070 | #18070 or a per-op `PREC_F32` fix lands upstream | `poison_probe`; the fp16 canary | #18070 open |
| **`903` MMQ ids-path padding** | llama.cpp#27044 | fixed upstream | the MoE + MMQ functional gate | #27044 open |
| **`801` clip node-stats meter** | #214 diagnostic | with the f32 gate above; it exists to localise that overflow | the fp16 canary under `OLLAMA_CLIP_NODE_STATS` | keep while the gate is carried |
| **`004` gemma4 budget fill** (`image_budget_fill`, `PAD_NONE`) | ADR 0008 | upstream fills to the ladder and stops letterbox-padding. Its default *limits* converged on ours (70/1120) in b10864 | `pinned_image_token_budget`, `token_ladder`, bbox conformance | limits converged; the fill is still ours |
| **`002` nemotron native-aspect dynamic resolution** | ADR 0001 | upstream leaves the fixed 512×512 canvas | `token_ladder`; nemotron bbox cells | fixed canvas upstream at b10864 |
| **ADR 0014's fork-local half**: `growOpeningChunk` and the all-or-nothing match for bidirectional expansions | ADR 0014 | upstream's non-causal boundary work (0.34.1) covers bidirectional expansions | `media_test.go`, the prefix-cache media tests, gemma4 multi-image cells | partly converged: `extendChunk` is upstream's |
| **MLX admission** (KV priced at the rung, per-arch headroom, refuse/clamp) and the **memory and cache limits** | ADR 0034 | upstream prices KV at admission and exposes the limits | `client_admission_test.go`, `client_budget_test.go`; the OOM ladders | upstream bounds by free memory only; it generates the limit symbols but does not call them |
| **stop sequences on MLX** | `stopper.go` | upstream's MLX runner honours `stop` | `stopper_test.go` | none upstream |
| **per-model `kv_cache_type`** | ADR 0005 | upstream adds a per-model option | `llm/kv_cache_type_test.go` | one global env upstream |
| **scheduler fixes**: `evictAbandoned`, head-of-line, leaf `logMu` | `server/sched.go` | fixed upstream | `sched_test.go`, `sched_headofline_test.go` | fork-only |
| **capability rules for MLX arches** (nemotron text-only, gemma4 without audio) | `server/images.go` | upstream's `filterUnsupportedCapabilities` handles them | `images_test.go` | fork-only, re-homed in the 0.34.1 fold |
| **gemma4 image chunk vs. batch**: the launcher's `-b N -ub N` keeps llama.cpp #28954's abort away, and since ADR 0036 the scheduler sizes the batch to the image ceiling so the non-causal chunk decodes in one piece | task doc 0.34.1, 2026-09-18 | upstream fits an image chunk to one ubatch (llama.cpp #28954's fix) or sizes ubatch to the image cap | the top-rung bbox cells at `num_batch` 1024 vs 2048; preflight `token_ladder` | #28954 open; **fixed in the fork 2026-09-18** (ADR 0036, #320): a gemma4 vision runner starts from the batch rung holding its image ceiling, measured first (no contract cost, 1.5–1.9× prefill, 31b 9 px tier 4 → 3) |
| **the drafting knob as memory mitigation** (`OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`, deployed 2026-09-18): the MLX runner retains 0.1–0.3 GiB per image request that ends by stop under speculation (component A, grammar or not); the knob removes it for grammar requests only | task doc 0.34.1 "drafting retention"; `upstream-mlx-speculation-image-stop-retention.md` (parked) | upstream finds and frees the holder on MLX's side | the size ladder (`leak-repro5.sh`): `held − trie` flat with drafting on | reproduced, exclusions measured, no fix known |
| **transparent images composited over white** (gemma4) | ADR 0015 | upstream's gemma4 path composites; its qwen3.5 path drops alpha by RGB conversion instead, per that model's reference | gemma4 vision goldens | not converged |

## Fork tooling with no upstream counterpart

| item | retires when | status |
|---|---|---|
| `Dockerfile.gemma4budget` + `docs/maxusai/gemma4-budget-image.md` | the overlay cannot carry payload patches, and every fold has built full images since 0.32 | stale; Glenn's call |
| `Dockerfile.applearm` + `docs/maxusai/spec/apple-silicon-build.md` | the Metal half is built another way, or dropped | held with the Metal half |

## Already retired

- **`x/structured`** (ADR 0009/0013), deleted 2026-09-17 in the v0.34.1 fold on Glenn's word, after the parity gate
  found 0 regressions against xgrammar v0.2.5 in 108 verdicts. Its one finding — `allOf` with several branches is
  permissive on xgrammar — is recorded in ADR 0033's amendment and pinned by
  `x/mlxrunner/xgrammar/engine_behaviour_test.go`, with the ADR 0013 bound kept there as a budget test.
- ADR 0017's mechanism (`mlx.ClaimOSThread`) — upstream's `mlxthread.Start` carries the guarantee since the 0.33.3 fold.
- ADR 0007 (gemma4 default budget 560) — superseded by ADR 0008.
- The integrated-GPU admission bound — upstream's, absorbed into `admit()` in the 0.34.1 fold.
- gemma4's default image-token limits — upstream adopted 70/1120 in b10864; our flags still pass them, per request.
