# Retirement register: what the fork carries, and what retires it

The README's promise is that every fork delta is "offered upstream where it is upstream's to take, and deleted from
here once it lands there". This is the list that makes that checkable at each fold. One row per carried item: what
retires it, and the test that must pass before it is deleted — deletion is a change like any other and gets a gate.
Reviewed at every fold; last against upstream v0.34.1 / llama.cpp b10864 (2026-09-17).

## Retire now, once tested

| item | record | retires when | test that gates deletion | status |
|---|---|---|---|---|
| **`x/structured`** — the pure-Go constrained sampler, 3,829 lines | ADR 0009, 0013 → 0033 | it has had no importers since ADR 0033 adopted upstream's xgrammar engine; ADR 0033 already names deletion as its follow-up | **Parity against upstream's engine, then delete if no regression (Glenn, 2026-09-17).** (1) Compile parity: every schema `x/structured`'s tests accept compiles on `x/mlxrunner/xgrammar`. (2) Bounded work: every schema ADR 0013 rejects for unbounded repetition (`maxRepetitionThreshold = 2000`, the cumulative product guard) either fails to compile on xgrammar or compiles inside a fixed time and memory budget — the property, not the rejection, is what ADR 0013 protects. (3) Acceptance parity: the JSON its tests accept is accepted by the xgrammar matcher, the malformed and incomplete inputs are rejected, and whitespace parity holds (`TestJSONWhitespaceParity`; see ADR 0035). (4) Live: the campaign's format and contract cells unchanged against the last run — those have measured upstream's engine on every fold since ADR 0033. Runs on CPU inside the built image, where `libollama_xgrammar.so` lives | carried; test not yet written |

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
| **transparent images composited over white** (gemma4) | ADR 0015 | upstream's gemma4 path composites; its qwen3.5 path drops alpha by RGB conversion instead, per that model's reference | gemma4 vision goldens | not converged |

## Fork tooling with no upstream counterpart

| item | retires when | status |
|---|---|---|
| `Dockerfile.gemma4budget` + `docs/maxusai/gemma4-budget-image.md` | the overlay cannot carry payload patches, and every fold has built full images since 0.32 | stale; Glenn's call |
| `Dockerfile.applearm` + `docs/maxusai/spec/apple-silicon-build.md` | the Metal half is built another way, or dropped | held with the Metal half |

## Already retired

- ADR 0017's mechanism (`mlx.ClaimOSThread`) — upstream's `mlxthread.Start` carries the guarantee since the 0.33.3 fold.
- ADR 0007 (gemma4 default budget 560) — superseded by ADR 0008.
- The integrated-GPU admission bound — upstream's, absorbed into `admit()` in the 0.34.1 fold.
- gemma4's default image-token limits — upstream adopted 70/1120 in b10864; our flags still pass them, per request.
