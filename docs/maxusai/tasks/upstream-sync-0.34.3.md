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
| 4, image | not built — a Go-only swap is valid for this fold (below) |
| 5, preflight | not run — see "Preflight profiles" for which surfaces resolve today |
| 6, campaigns | not run |
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
