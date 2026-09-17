# TASK: fold upstream v0.34.1 into main

Upstream [v0.34.1](https://github.com/ollama/ollama/releases/tag/v0.34.1) (tag `38fdb5dd5`, 2026-09-14; 23 first-parent
commits; 346 files, +17k/−57k) folded on top of `v0.34.0-dynres`. Branch `task/upstream-sync-0.34.1`, worktree
`claude-scratch/wt-sync0341`. Glenn said go on 2026-09-17, with #300 merged first and #301's ADR renumbered to 0035.
Dry-run conflict list: `claude-scratch/sync0341-dryrun-conflicts.txt`.

## Status (2026-09-17)

| gate | state |
|---|---|
| 1, merge | conflicts resolved (10 files, below); not yet committed |
| 2, no-GPU tests | in progress: build clean after the bench port; vet, tidy, `x/mlxrunner`, `llm`, `server` tests running |
| 3, image | **first attempt failed at 3.5 min** — patch 004 no longer applied to b10864 (below); re-cut, second build running. A native rebuild is required: llama.cpp, MLX and MLX-C all moved, so no binary swap is valid |
| 4, preflight `cuda-dynres-903` | pins updated below; run pending the image |
| 5, campaigns | pending the image: GGUF think-off vs `ggml034_1_1_`, MLX think-off vs `cand034_`, and the qwen2.5vl six-cell run as a second GGUF reference |
| memory re-measure | pending the image: the Pin/Sweep trace method is gone; `held` per request replaces it (below) |

## What v0.34.1 changes for the fork

- **All three pins move:** llama.cpp `b10760 -> b10864`, MLX `ce916dbb -> d9add9d1`, MLX-C `c74db530 -> ebc88f10`.
  Our `ce916dbb` (the idle-core fix, ml-explore/mlx#4452) is an ancestor of `d9add9d1`, 24 commits back, so upstream's
  bump carries it. The `#4458` `gather_qmm global_scale` wall that capped our MLX pin is gone: upstream hit it too
  and ships `mlx/compat/0001-mlx-c-qmm-global-scale.patch`, applied through `cmake/apply-git-patches.cmake` from
  `cmake/local.cmake`, and the Dockerfile's mlx stage now copies `mlx/compat`. We never changed `cmake/local.cmake`,
  so that merged clean. **The Go-only swap-validity input list gains `mlx/compat/`.**
- **The array-lifetime model is replaced** (`13037ecb1`, "scope array lifetimes instead of pinning and sweeping").
  `Pin`, `Unpin`, `Sweep`, `LogArrays` and the `tensors total:` trace line are gone; `x/mlxrunner/mlx/scope.go` adds
  function scopes (`Scoped`, `ScopedEval`, …) and held scopes (`NewScope`/`Attach`/`Discard`/`Close`). The runner
  logs one `memory` line per request: `peak`, and `held` — MLX's active memory after the request's scope ended and
  the cache was cleared. Upstream also releases the buffers weight loading leaves in the pool (`b68b112bd`).
- **`convert/` is deleted** (74 files → 0, "drop GGUF conversion"). Upstream's note: GGUF creation now needs
  llama.cpp's tooling. Our nemotron3 Omni converter fix (`147930aa2`) goes with it; nothing at runtime depends on it.
- **Capabilities and `/api/tags` rewritten** (`ea8d65004`): `server/model_list_cache.go` deleted; capabilities are
  computed in `images.go`'s `filterUnsupportedCapabilities`, which the auto-merge already carries our
  Nemotron-text-only and gemma4-safetensors rules into.
- **Drafting under a grammar stays on upstream** (`enabled := !opts.Logprobs && opts.TopLogprobs == 0`), so
  `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR` (ADR 0033) stays as it is; the merge kept `draftingEnabled` intact.
- Also: `typical_p` deprecated; the built-in agent removed; the repeat-token limit raised to 100 and now an error
  rather than a silent close; gemma3n projector kept off the CPU; three prefix-cache eviction fixes; "bound MLX
  loads by system free memory while other models are loaded"; the scheduler waits for a killed runner before the
  next load.

## Conflicts and their resolution

| file | ours vs v0.34.0 | upstream | resolution |
|---|---|---|---|
| `MLX_VERSION` | `ce916dbb` | `d9add9d1` | upstream's; it contains ours |
| `convert/convert_nemotron_h.go` + test | +5/−1 | deleted | deletion accepted (D3) |
| `server/model_list_cache.go` | +4/−1 | deleted | deletion accepted; the rule already lives in `images.go` |
| `llm/llama_server.go` | +680/−74 | +51/−44 | one hunk: upstream's repeat-limit error in our `(result, error)` arity |
| `x/mlxrunner/client.go` | +356/−35 | +25/−23 | six hunks: our admission signature and fields plus upstream's `closed` flag, `CheckRuntime`, and the locked closed-check before `Start`; upstream's integrated-GPU clamp folded into `admit`, which now takes `systemInfo` (10 test callers updated) |
| `x/mlxrunner/runner.go` | +257/−3 | +95/−79 | model setup moved to upstream's `loadModel`; only `configureCacheLimit()` kept, after the weights are resident |
| `x/mlxrunner/pipeline.go` | +58/−11 | +106/−102 | teardown: our `guardClose` discipline on upstream's scoped shape, logging `memory peak/held`; streaming: upstream's scoped loop already records-before-streaming, so our stop-sequence handling was ported into it (`stop.feed`, empty-chunk skip, `stopMatched` break) |
| `server/routes_debug_test.go` | +12/−6 | +5/−5 | our `vision.block_count` fixture lines on upstream's `gguftest` types |
| `server/routes_generate_test.go` | +1617/−34 | +100/−35 | our `newTensors()` helper on `gguftest` types; `ggml.KV` is gone at v0.34.1, so the four residual fixtures migrated and the unused import dropped |

Auto-merged and to be proven by tests rather than read: `speculate.go` (knob intact), `prefix_cache.go`, `sched.go`
(our `logMu` hunks and upstream's killed-runner wait do not overlap), `Dockerfile`, `test.yaml` (our Darwin cache
key survived), `media.go`, `images.go`.

## Gate 3: the patch series against b10864

The first build died at 3.5 minutes in the vulkan llama-server stage: `004-llama-cpp-gemma4-budget-fill.patch`
no longer applied. Between b10760 and b10864 upstream changed the gemma4 projector case's
`set_limit_image_tokens(40, 280)` to `(70, 1120)` and dropped the comment above it — both were context lines of our
`clip.cpp` hunk. The insertion point itself (after the case's warmup line) had not moved.

Checked the whole series on a real b10864 checkout (`claude-scratch/llama.cpp-src`, blobless clone), applied in build
order with plain `git apply`: 001, 002, 005, 801 and 903 apply at offsets; only that one hunk of 004 failed. Re-cut
it by anchor and regenerated 004 from `git diff`: the added and removed lines are identical to before (+67/−8), only
context and offsets differ, and the six-patch series then applies clean from a clean checkout.

**Lesson, recorded so the next fold does not repeat it:** when a pin moves, the patch series must be applied to a
checkout of the new pin before the build. The fork's own patch files being unchanged between two tags — which this
fold checked and even published for 0.34.0 — says nothing about whether they still apply. The 0.33.3 fold did the
checkout check; this one skipped it and paid 3.5 minutes, which is cheap only because the vulkan stage runs first.

**To watch in gate 4:** upstream's new gemma4 limits, 70 and 1120, are exactly the endpoints of the budget ladder
our `image_budget_fill` was written for. `pinned_image_token_budget` and `token_ladder` will say what the served
budgets do under the two together.

## Ports off the removed lifetime API

Three files of ours used it; everything else was upstream-owned and came rewritten.

- `x/mlxrunner/bench/qqmm/main.go` (11 sites): each shape's weights live in a function scope, each `m`'s inputs in a
  nested one, and `measure` wraps every timed call and the error check in its own scope, so nothing is pinned or
  swept by hand. `QuantizedMatmul` also gained a trailing global-scale argument with the carry patch.
- `vision_e2e_test.go`, `vision_golden_test.go` (1 each): the worker's stop hook no longer sweeps; the runner's
  arrays live in held scopes that `Close` releases.

## Decisions

- **D1 — take upstream's pins and its MLX-C carry patch as they are.** The ceiling we recorded ("cannot go past
  #4458 without an MLX-C bump") is lifted by upstream's own patch; we carry nothing extra.
- **D2 — port to scoped lifetimes rather than reintroduce Pin/Sweep.** Only three of our files needed it.
- **D3 — accept the deletion of `convert/`.** `ollama create` from safetensors to GGUF is llama.cpp's job upstream
  now; the store holds prebuilt GGUFs and the teacher pipeline never converts.
- **D4 — the memory measurement moves to `held`.** `runnerlog.py` reads the new `memory` line as a request boundary;
  `summarize_retained_memory.py` prints `held` and its step between requests on such logs (two tests). The
  drafting-leak finding of the v0.34.0 fold was of the old model and is re-measured on this build, not carried.
- **D5 — the knob stays.** Upstream still drafts under a grammar; ADR 0033 is unchanged.

## Fork against upstream v0.34.1

The capability-level table — what the fork does that upstream does not, one row per capability, measured against
upstream v0.34.1 at llama.cpp b10864 — is the README's "What differs from upstream, concretely", rewritten in this
fold. This section only sizes the divergence for the merge: `git diff v0.34.1 task/upstream-sync-0.34.1` is 425
files, +91,641/−2,696; outside `docs/maxusai`, 123 files, +21,874/−2,696. The merge-base with `upstream/main` is the
tag itself, so all of it is fork-authored. The largest non-docs areas are `x/mlxrunner` (23 files), `server` (11),
`x/structured` (11, +3,829 — **no importers since ADR 0033; dead code, Glenn's call**), `x/models/gemma4` (10, our
vision with upstream's tower excluded), `x/mlxrunner/kvsize` (9), `llama/compat` (6 patches), `llm` (4).

## The gemma4 image-token limits (Glenn's question, 2026-09-17)

llama.cpp `163a40796` — "model, mtmd: fix gemma4 vision handling" (#28335, 2026-09-04, in b10864) — changed the
gemma4 projector's default `set_limit_image_tokens(40, 280)` to `(70, 1120)` and dropped the comment above it. That
is the change that broke patch 004's context. The same commit also touched the text side (`llama-hparams.h`,
`llama-kv-cache.cpp`, `models/gemma4.cpp`).

- **On this fork it is inert.** Every gemma4 llama-server runner receives `--image-min-tokens` and
  `--image-max-tokens` from `gemma4ImageTokenBudget`, defaulting to `api.DefaultImageMinTokens = 70` and
  `DefaultImageMaxTokens = 1120` (ADR 0008), and llama.cpp's `set_limit_image_tokens` yields to those
  (`custom_image_min/max_tokens` win when set). Our served budget was already 70/1120; upstream's default has now
  converged on it — for stock ollama users the gemma4 default ceiling quadrupled.
- **What the fork still carries beyond upstream** is 004's `image_budget_fill` and `PAD_NONE`: snap the grid to a
  ladder rung and fill it, never letterbox. Upstream raised the ceiling; it did not adopt the fill.
- **Follow-up after the image is gated:** the comment in `llm/llama_server.go`'s gemma4 branch still says
  "llama.cpp defaults to set_limit_image_tokens(40, 280)". Left for now so the image's Go code stays identical to
  the branch head while the gates run.
- **Watch in gate 4:** other commits in the b10760..b10864 window reworked `mtmd-image.cpp` (+204/−32); 004's
  fill hunks apply there at offsets. `pinned_image_token_budget` and `token_ladder` will show whether the served
  grids still land on the ladder.

## Retirement candidates (Glenn, 2026-09-17)

`x/structured` stays until upstream's engine is tested against it; if no regression, it is deleted. That decision,
the parity test it needs, and every other item the fork carries with its retiring condition and gate are in
[`docs/maxusai/retirement-register.md`](../retirement-register.md), reviewed at each fold. Two rows moved in this
one: upstream 0.34.1 has its own transition-based `format` deferral with pass-one metrics (ours is now a superset,
README row corrected), and `extendChunk` from ADR 0014 turns out to be upstream's already.

## Not in this fold

- The Metal half: MLX and MLX-C moved, so the Metal payload changes too; held by Glenn.
- `v0.34.2-rc1` (8 commits further) moves the MLX engine out of `x/` and re-lays out the models; every fork path
  under `x/mlxrunner` re-homes. That is a separate, structural fold.
- PR #301 (Glenn's whitespace bound, renumbered to ADR 0035, green): merging it before this branch lands means a
  small merge of `client.go` here; after, it rebases onto the resolved file. Either is fine.
