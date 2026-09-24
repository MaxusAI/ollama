# TASK: fold upstream v0.34.3 into main

Upstream [v0.34.3](https://github.com/ollama/ollama/releases/tag/v0.34.3) (tag `6383a0fa9`, 2026-09-18; six commits;
68 files, +3,896/−355; 13 added, 55 modified, none renamed or deleted) folded on top of `v0.34.2-dynres`. Branch
`task/upstream-sync-0.34.3`, worktree `.claude/worktrees/upstream-sync-0343` on the gfx1151 host, cut from `main` at
`c86c5851` (#370).

## Status (2026-09-24)

| gate | state |
|---|---|
| 1, the merge | **done** — `0fcce2a4`, one conflicted file (`server/routes.go`, four hunks); `go build` and `go vet` clean, `go test` 58 packages ok |
| 2, docs and paths | **nothing to re-point** — the fold renames and deletes nothing; `check_source_paths.py` clean |
| 3, the patch series | **not applicable** — llama.cpp `b10969`, MLX `d9add9d1`, MLX-C `ebc88f10`, all identical to `main` |
| no-GPU harness gates | **green** — `test_verdicts.py` 193 OK (6 skipped), `test_summarizers.py`, `test_rescore.py`, `test_mlx_test_gate.py` |
| 4, image | **gfx1151: built** — `Dockerfile.rocm` (ADR 0042) on `rocm/dev-ubuntu-24.04:7.2.4-complete`, gfx1151, payload gated against production. CUDA and Metal: not built |
| 5, preflight | not run — rocm7 has no `b10969` profile (see "Preflight profiles"); CUDA and Metal not run |
| 6, campaigns | **gfx1151 think-off: done, no regression** — every scored cell of five models and every OCRBench item equal to production (below). Think-on running. CUDA and Metal not run |
| tag and deploy | not done; `v0.34.3-dynres` is cut on the merge that lands this, per ADR 0032 |

The MLX-dependent tests skip on the gfx1151 host, which has no MLX library, so "58 packages ok" covers the Go and
GGUF paths only. The macOS leg of `test.yaml` prepares the MLX Darwin payload and runs them there; that CI run is the
first execution of upstream's new nemotron_h vision tests on this tree.

## What v0.34.3 changes for the fork

- **Model thinking levels** (#18473, `d0c8cdb79`). Renderers declare the levels they accept and a default,
  `/api/show` reports them as `thinking`, and the chat, generate and OpenAI/Anthropic-compatible routes resolve an
  omitted or unsupported level to the declared default (`renderers.ResolveThinking`). `ThinkValue` moves to
  `types/model` behind an alias in `api`, and any string now counts as "thinking on" in `Bool()`. That is safe for
  every renderer here: each spells "off" as the bool `false`, and every string level it declares (low, medium, high,
  max, xhigh) means on. **Omitting `think` keeps its old meaning** on thinking-capable local models:
  `Model.Thinking()` lifts a declared `false` default to `true` wherever `true` is accepted, and qwen3.8 resolves to
  `medium`, the level `true` already reached. gemma4's renderer declares `Default: false`, and the lift is what keeps
  it default-on.
- **nemotron_h vision on MLX** (#17714, `2c7316424`). The RADIO tower and projector, native-aspect dynamic resolution
  between the checkpoint's `min_num_patches` and `max_num_patches` (1024…13312 for the Omni `c-radio_v4-h` config in
  upstream's tests: 256…3,328 tokens after the 2×2 shuffle, the same bounds our GGUF `002` sets), and the capability
  flip that stops suppressing vision for nemotron safetensors.
  That suppression was upstream's own rule (`suppressVisionCapability`, #17060). The fork's copy went with
  `model_list_cache.go` in the 0.34.1 fold, and the fork's `images.go` delta is only the gemma4-audio rule (ADR 0021).
- **Registry redirects among allowlisted hosts** (#18533): `ollama.com`, `ollama.ai`, `hf.co`, `huggingface.co` and
  their subdomains may redirect to each other; everything else stays same-host only unless `--insecure`.
- **Release artifact uploads hardened** (#18516), a metadata-deletion test flake fixed (#18493), a macOS app window
  fix (#18518).

## The conflict and its resolution

**`server/routes.go`, four hunks, one cause.** Upstream's thinking-levels change adds a local `thinking :=
m.genericThinking()` to both handlers, which shadows the `thinking` package, so upstream imported the package as
`thinkingparser` and renamed `thinkingState` to `thinkTagParser`. Every hunk is that rename landing on the fork's
ADR 0004 marker flow:

| hunk | ours | upstream | resolution |
|---|---|---|---|
| generate, parser setup | `thinkCloseTag` from `ImplicitThinkingParser`, then `var thinkingState *thinking.Parser` | `var thinkTagParser *thinkingparser.Parser` | ours, under upstream's names |
| generate, opening tag in the prompt | `AddContent(openingTag)` and `thinkCloseTag = closingTag` | `AddContent(openingTag)` | ours, under upstream's names |
| generate, stream callback | the pass-one marker block, then our own `thinkingState` branch | upstream's shorter `thinkTagParser` branch at the old position | ours; our branch is the superset and follows below it |
| chat, format deferral | `deferring := false` and `formatConstrains(req.Format)` | `req.Format != nil`, renamed parser | ours, under upstream's names |

The eight fork-only references to `thinkingState` outside the hunks were renamed the same way. Upstream's own changes
in the file — the raw `think` binding with `thinkingInputError`, `ValidateLegacyThinking`, the parser `Init` with the
resolved level, `Thinking` in the show response, `lookupThinking` in the compatibility middleware — merged clean.

**The carry is exact.** The fork's `routes.go` delta is +691/−87 against v0.34.2 before the fold and +691/−87 against
v0.34.3 after it, line for line once the two renames are applied. The same comparison over all 488 files in the fork's
delta finds no other difference. Three files flagged on the first pass — two docs and `routes_generate_test.go` —
mention `thinking.InferTags` in prose, which is still the package's own name; they are unchanged.

## Auto-merges read rather than trusted

| file | ours | upstream | why it holds |
|---|---|---|---|
| `server/images.go` | gemma4 safetensors suppresses audio (ADR 0021) | nemotron safetensors keeps vision; `isNemotron3NanoSafetensors` → `isNemotronSafetensors`, widened to nemotron-3.5; redirect allowlist | disjoint functions |
| `server/images_test.go` | three gemma4 audio cases | nemotron3 and nemotron3.5 expose vision | both sets pass |
| `api/types.go` | `ImageMin/MaxTokens`, `KVCacheType`, their defaults | `ThinkValue` and `ModelRecommendationThinking` become aliases into `types/model` | disjoint |
| `model/parsers/nemotron3nano.go` | `ThinkingCloseMarker()` for the marker flow | flush buffered thinking on `done` | different methods; the marker flow feeds the parser with `done=false` |
| `.github/workflows/release.yaml` | the fail-fast guard in `setup-environment` | upload hardening | every job still needs `setup-environment`, so the guard still stops a runnerless release |
| `mlxrunner/model/nemotron_h/nemotron_h_test.go`, `model/parsers/nemotron3nano_test.go` | ours | 281 and 93 lines of new tests | compile and pass |

## Gate 4: a Go-only swap is valid here, minus one inert patch

The 0.34.0 fold learned that identical pins are not enough: upstream changed `mlxrunner/xgrammar/native`, which ships
inside the MLX payload, and a Go-only swap answered every MLX `format` request with HTTP 501. This fold changes no
native input at all. Against `main`, nothing moves under `LLAMA_CPP_VERSION`, `MLX_VERSION`, `MLX_C_VERSION`,
`llama/`, `ml/`, `mlx/compat/`, `mlx/CMakeLists.txt`, `cmake/`, `CMakeLists.txt`, `CMakePresets.json`, `Dockerfile` or
`mlxrunner/xgrammar/native`, and the only non-Go files in the diff are `release.yaml` and the desktop app's
`app/cmd/app/app_darwin.m`.

`main` itself has moved since both deployed payloads were built: #370 added `llama/compat/802-lm-node-stats-meter.patch`
after `0.34.2-dynres-0-g5bffaac` (CUDA) and `0.34.2-dynres-f67b1aef` (gfx1151). That is the only native difference —
802 and its README entry. 802 is gated on `OLLAMA_LM_NODE_STATS` and falls through to upstream's eval callback when the
variable is unset, so a Go-only swap onto either deployed payload serves what a full build would serve, and differs
from one only in lacking a meter nobody has switched on. A build that needs 802 must be a full build.

## Preflight profiles

Resolved with the harness's own `resolve_profile` against the stamps this fold would carry:

| surface | stamp | profile |
|---|---|---|
| cuda | `0.34.3-dynres-0-g…` | `cuda-dynres-903` — the lineage pattern admits `0.3[234]`, and its `b10969` payload pin is unchanged |
| mlx-cuda | `0.34.3-dynres-0-g…` | `mlx-cuda` — admitted; its arches are unmeasured, so it exits 4 as before |
| mlx-metal | `0.34.3-dynres-0-g…` | **none** — `mlx-metal-0-34-2` admits 0.34.2 only, "widen deliberately". The payload does not move, so ADR 0032 says widen rather than cut; that is the Metal host's step, after its own run |
| rocm7 | `0.34.3-dynres-…` | **none** — and today's production stamp `0.34.2-dynres-f67b1aef` resolves to none either. The only 0.34 rocm7 profile is `rocm-0-34-1-dynres`, pinned to `b10864` with 906; `b10969` on rocm7 has never had one. The 0.34.2 promotion on 2026-09-21 went through the upgrade gate's clauses instead ([amd-upgrade-gate.md](../amd-upgrade-gate.md)). This gap predates the fold; a measured `b10969` rocm7 profile would serve 0.34.2 and 0.34.3 alike |

No profile is edited here. Profiles are measured on the host that serves them and land in their own change, as
`rocm-0-34-1-dynres` did (#344).

## Gates 4 and 6 on gfx1151 (2026-09-24)

**No regression.** The candidate scores exactly like production on this host: every scored cell of the five
on-host vision models, and the same verdict on every one of 200 OCRBench items.

### What was compared

| arm | image | built | served on |
|---|---|---|---|
| control | `maxusai-ollama:0.34.2-rocm724-main-f67b1aef` — production, `0.34.2-dynres-f67b1aef` | upstream's `Dockerfile`, `FLAVOR=rocm`, AlmaLinux 7.2.4, all 13 targets | `:11499` |
| candidate | `maxusai-ollama:0.34.2-dynres-24-gef19770-rocm7-gfx1151` — this branch at `ef197701b` | `Dockerfile.rocm` (ADR 0042, #374), Ubuntu 7.2.4, gfx1151 | `:11499` |
| ROCm 10 | `maxusai-ollama:0.34.2-dynres-25-g3e9bc1f-rocm10-gfx1151` — this branch plus the `rocm_v10_0` presets | `Dockerfile.rocm`, Ubuntu 10.0.0, gfx1151 | `:11498` |

Every arm ran in a canary container with production's environment (`OLLAMA_FLASH_ATTENTION=1`, `q8_0` KV,
`OLLAMA_NUM_PARALLEL=2`), plus `OLLAMA_MAX_LOADED_MODELS=1` and `OLLAMA_NOPRUNE=1`, and a cold container per
model. Production on `:11434` was not touched. The suite is `run_engine_compare.sh` at its defaults (chat
endpoint, `num_ctx` 16384, `num_predict` 2200, greedy, think off), over the five clause-4 models. OCRBench is
rows 0–200, `gemma4:31b-it-q4_K_M`, think off, `OLLAMA_NUM_PARALLEL=1` (the ROCm ladder's placement).

**The candidate carries this fold and the ADR 0042 toolchain together.** Its compiler is the same build as
production's: the device code in both reads *AMD clang 22.0.0git roc-7.2.4 26084 f58b06dc*. The payload gate
against production found 0 SONAMEs missing and 96 = 96 gfx1151 rocBLAS kernel files, and `llama-server` reports
`commit 391fac164`, the pin preflight records. The candidate is gfx1151-only: that GPU's device code is compiled
independently of the other targets, so this is the code a full build would run here, at a twelfth of the HIP
compile. A general-purpose deployable wants the full target list.

**The candidate's payload is ccache-built, and so is production's.** Its HIP objects were ccache hits,
and the whole candidate build took about three minutes. To see what that means, the ROCm stage was compiled twice
more with `CCACHE_DISABLE=1` (the ccache counters did not move either time). Those two clean compiles produced
byte-identical `libggml-hip.so`, so the build itself is deterministic. The ccache-served library differs from them
in every one of its 143 HIP compilation-unit IDs (`__hip_cuid_*`, which clang derives per compile), and in about
116k further bytes that plausibly follow from them. Code objects (137), file size, section layout and the compiler
build are identical, and the artifact scores exactly like production below. A cached build is therefore
functionally reproducible against a clean one, not bitwise. Production's AlmaLinux images went through the same
ccache wiring, so this predates ADR 0042. Run: `nocache-verify/` in the run directory.

### Results

| comparison | vision suite, scored cells differing | OCRBench, discordant items |
|---|---|---|
| control vs production's recorded clause-4 run (`gate4_0342`, 2026-09-20) | **0** of 978 / 976 / 986 / 980 / 983 | — |
| **candidate vs control** | **0** of 978 / 976 / 986 / 980 / 983 | **0** of 200 (172 = 172) |
| ROCm 10 lane vs the 0.34.2 ROCm 10 probe (`rocm10`, 2026-09-20) | **0** of 978 / 976 / 986 / 980 / 983 | 0 vs control (172 = 172) |

Models in column order: qwen3.6:35b-a3b, qwen3.8:27b, gemma4:31b, gemma4:26b-a4b, nemotron3:33b, all q4. Cells
are every scalar score field of every block (`cmp_scores.py`, in the run directory), including
`prompt_eval_count`, `eval_count` and `done_reason`. Provenance and timing fields are excluded.

The first row is what makes the rest mean something. A fresh control on the production image reproduced a
four-day-old run of that image in every scored cell, so GGUF think-off on this host is deterministic across
days, containers and GPU contention. A zero difference between candidate and control is therefore a
measurement, not a coincidence of noise.

The candidate, rendered by `summarize_engine_compare.py` (verbatim). The control's quality columns are
identical.

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| qwen3.6:35b-a3b-q4_k_m | GGUF | 16384 | 0.966 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 0.984 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 0.965 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384 | 0.975 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 0.862 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.6:35b-a3b-q4_k_m | GGUF | 16384 | 4 | 4 | 4 | 2 | 2 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 544 | 19 | 303 | 36.7 | 98 |
| qwen3.8:27b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 3 | 1 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 544 | 6 | 142 | 107.4 | 34 |
| gemma4:31b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 538 | 8 | 783 | 70.3 | 51 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 536 | 25 | 2027 | 22.2 | 162 |
| nemotron3:33b-q4_K_M | GGUF | 16384 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 512 | 37 | 240 | 24.8 | 145 |

Provenance (from score files): host(s) http://127.0.0.1:11499 · build(s) 0.34.2-dynres-24-gef19770 · think=false

**Throughput columns are not comparable across arms.** The candidate arm shared the iGPU with the ROCm 10 lane
for its whole run; the control shared it for part of its run. Quality is unaffected, because decoding is
greedy, as the zero differences show.

OCRBench, rendered by `summarize_extbench.py --paired --categories --timing` (verbatim). The MIXED banner is
correct: three builds on two ports is the point of the table.

| model | scored | errors | empty | correct | accuracy | think | endpoint |
|---|---|---|---|---|---|---|---|
| `gemma4:31b-it-q4_K_M` | 200 | 0 | 0 | 172 | **0.86** | false | generate |
| `gemma4:31b-it-q4_K_M` | 200 | 0 | 0 | 172 | **0.86** | false | generate |
| `gemma4:31b-it-q4_K_M` | 200 | 0 | 0 | 172 | **0.86** | false | generate |

ocrbench — `echo840/OCRBench` [test], rows 0..200.

⚠ **MIXED — rows are not one campaign** (hosts: ['http://127.0.0.1:11498', 'http://127.0.0.1:11499']; builds: ['0.34.2-dynres-24-gef19770', '0.34.2-dynres-25-g3e9bc1f', '0.34.2-dynres-f67b1aef'])

| pair | both ✓ | both ✗ | A only | B only | McNemar exact p |
|---|---|---|---|---|---|
| r0343ctrl_ocr_q4 vs r0343cand_ocr_q4 | 172 | 28 | 0 | 0 | 1.000 |
| r0343ctrl_ocr_q4 vs r0343r10_ocr_q4 | 172 | 28 | 0 | 0 | 1.000 |
| r0343cand_ocr_q4 vs r0343r10_ocr_q4 | 172 | 28 | 0 | 0 | 1.000 |

| arm | accuracy | ±1 s.e. | mean s/item | median | mean prompt_eval |
|---|---|---|---|---|---|
| ctrl | 0.860 | 0.025 | 10.1 | 10.1 | 1115 |
| cand | 0.860 | 0.025 | 10.3 | 7.9 | 1115 |
| r10 | 0.860 | 0.025 | 13.6 | 13.4 | 1115 |

| question type | n | ctrl | cand | r10 |
|---|---|---|---|---|
| Artistic Text Recognition | 50 | 48/50 | 48/50 | 48/50 |
| Handwriting Recognition | 50 | 34/50 | 34/50 | 34/50 |
| Irregular Text Recognition | 50 | 41/50 | 41/50 | 41/50 |
| Regular Text Recognition | 50 | 49/50 | 49/50 | 49/50 |

### Think-on (running)

`thinkon.sh` interleaves control and candidate per model at a fixed 16384 window (`ALLOW_NO_LADDER=1`,
`CTX_MAX=16384`), greedy (ADR 0029). A cell that caps is capped identically on equivalent builds, so it stays a
like-for-like A/B, but it does not measure the window a model needs or its tok/s. ADR 0025 keeps think off in
production here; this arm exists because every model comparison runs both modes.

Run directory, with scripts, logs and score files: `/opt/github/MaxusAI/bench-0343/` on the gfx1151 host.

## Open items

1. **nemotron_h on MLX does not honour the per-request image budget.** It implements `model.MediaModel` but not the
   fork's `model.MediaBudgetModel`, so `mlxrunner/media.go` takes its fallback: `image_min_tokens` and
   `image_max_tokens` are logged as a warning and the model's own bounds apply. The GGUF path honours both for the
   same architecture through `002`. The fallback comment already names this case ("upstream models added between
   merges rather than a supported state"). Closing it means a `PrepareMediaWithBudget` that maps tokens to patches
   (×4 for the 2×2 shuffle), clamps to the model's `min/max_num_patches`, and puts the resolved budget in cache
   identity, as glimmer and qwen3.5 do. It can only be verified on an MLX host, so it is not done in this fold.
2. **README fold pointer.** It still names `v0.34.1-dynres` and a 0.34.1 deploy, stale since the 0.34.2 fold. Update
   it with `v0.34.3-dynres` and a matrix generated from that fold's full preflight run, as `#273` did after `#264`.
3. **Gates 4–6 and the tag**, on the hosts that own them.
4. **v0.34.4 is out** (2026-09-23) and is a different kind of fold: llama.cpp `b10969` → `b11081` and MLX
   `d9add9d1` → `59d600b5`, so gate 3 (the patch series on a real `b11081` checkout) and new payload profiles apply.
