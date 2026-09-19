# TASK: fold upstream v0.34.1 into main

Upstream [v0.34.1](https://github.com/ollama/ollama/releases/tag/v0.34.1) (tag `38fdb5dd5`, 2026-09-14; 23 first-parent
commits; 346 files, +17k/−57k) folded on top of `v0.34.0-dynres`. Branch `task/upstream-sync-0.34.1`, worktree
`claude-scratch/wt-sync0341`. Glenn said go on 2026-09-17, with #300 merged first and #301's ADR renumbered to 0035.
Dry-run conflict list: `claude-scratch/sync0341-dryrun-conflicts.txt`.

## Status (2026-09-17)

| gate | state |
|---|---|
| 1, merge | done: `f33d1888d`, 10 conflicted files resolved (below) |
| 2, no-GPU tests | green locally and on CI (13 jobs, race on both platforms) at `b1db10efc` |
| 3, image | built 15:31 (4 h 00 m, second attempt) as `maxusai/ollama:sync-0.34.1`, stamp `0.34.0-dynres-6-gfb18f5c`; the first attempt failed at 3.5 min on patch 004 (below) |
| 4, preflight `cuda-dynres-903` | **PASS 21 / SKIP 7** from the fold worktree, same as the deployed image |
| 5, campaigns | GGUF: no quality cell regressed, e2b and e4b recovered (e2b at n = 4), e2b's anchored-cell loop and 26b-a4b's two contract flips reproduce 4/4; qwen2.5vl: every quality cell identical; MLX (re-run on a free GPU): every scored cell equal; the one 35b-a3b cell that moved is bimodal across cold loads on the new image (0.504/0.613, n = 4) with the deployed value one of the modes — no regression |
| memory re-measure | done: under drafting + grammar the fold retained ~3× the deployed build (35b-a3b +5.3 vs +1.8 GiB over 28 requests, outside the trie); upstream's `ec3cc2307` brings it to +2.4 and is cherry-picked; the knob takes it to +0.1 and stays the mitigation for what both builds share (below) |

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

## Gates 4 and 5: the image on GPU0 (2026-09-17)

Image `maxusai/ollama:sync-0.34.1`, stamp `0.34.0-dynres-6-gfb18f5c`, built in 4 h 00 m on the `bigdisk` builder
(state on the array; only the final `--load` touches the Docker root). Chain `claude-scratch/gate-sync0341.sh`,
15:31–18:08: one container at a time on GPU0 beside production, 16 GiB overhead, cold restart per cell, never
`:11497`. GPU0 was shared for the whole chain with two foreign LoRA benchmarks (19 GB each, ~50 GB used in total,
60–90 % utilisation); that decides two things below.

**Gate 4, preflight from the fold worktree:** PASS 21, SKIP 7 — the same as the deployed 0.34.0 image. Version
matches the `cuda-dynres-903` profile, the measured llama.cpp payload is `5d806aa25`, the marker is in the binary.
Gate 3's watch item answered: under upstream's new (70, 1120) defaults and our flags together,
`token_ladder [gemma4]` is 5/5 geometries within ±2 and `pinned_image_token_budget [gemma4]` pins 560 → 529 with
ceiling 1120 — the served budgets are unchanged. (`preflight-runs/full-0341.{log,json}`.)

**Gate 5a, GGUF think-off, eight models, `ggml0341_1_1_` against `ggml034_1_1_`** — render
`preflight-runs/ggml0341-render-thinkfalse.md`, generators only:

- 31b, 27b, 35b-a3b, nemotron q4 and q8: every quality cell equal within 0.006 IoU; nemotron q8's name_bbox IoU
  0.044 → 0.165.
- e4b and e2b, the two the 0.34.0 gate flagged as regressions, recovered. e2b: scene IoU 0.061 → 0.496, invoice
  1/5 → 5/5, name_bbox in-band 0/5 → 4/5, fine text 0/0/0/0/0 → 4/4/3/1/0. e4b: IoU 0.354 → 0.462, invoice total
  ❌ → ✅, name_bbox 0.000 → 0.697, 9 px and 7 px 0/0 → 1/1, multi-image q2 ❌ → ✅. The only change on their
  path is the llama.cpp payload, b10760 → b10864; which upstream change did it is not isolated. The veto on
  "recovered" is a repeat of the 0.34.0 baseline showing its zeros were not a one-off.
- One cell errored: e2b `multi_3img_anchored`, HTTP 500 "prediction aborted, token repeat limit reached". That is
  upstream's 0.34.1 change in `llm/llama_server.go`: the repeat limit went 30 → 100, and a tripped limit now returns
  an error where it returned `ctx.Err()` — nil — so on 0.34.0 the same loop passed as a silently truncated HTTP
  200. The baseline passed this cell, so e2b looped on this prompt where it did not before; n = 1.
- Contract flags (`contract_followed`): +4 / −2. Gains: 31b `bcpinned`, e4b `bcanchored` and `bcadvnorm1`,
  nemotron q8 `bcreasoning`. Losses: 26b-a4b `bcreasoning` and `bcpinned`. n = 1.
- The throughput columns are not read: the campaign shared GPU0 with the LoRA jobs, and the numbers move both ways
  (35b-a3b 27 → 67 tok/s, 27b 44 → 30). A quiet-GPU repeat is the only way to compare them.
- Repeats for the two n = 1 movers, `ggml0341r_{1,2,3}_` (e2b and 26b-a4b, full suites, same image and settings,
  20:40–21:03; render `preflight-runs/ggml0341-reps-render.md`): **both reproduce 4/4 and every cell is identical
  across the four reps** — GGUF think-off is deterministic here, as ADR 0012 §4 says. e2b's recovery holds (IoU
  0.496, invoice 5/5, fine text 4/4/3/1/0 in all four) and its `multi_3img_anchored` loop trips the repeat limit in
  all four. 26b-a4b's `bcreasoning` and `bcpinned` are ❌ in all four, its scored cells unchanged from the 0.34.0
  fold. Both are real, deterministic changes of model output on the b10864 payload, not noise; neither is a
  regression in a scored cell. Remaining veto on "e2b recovered": the 0.34.0 baseline is n = 1.

**Gate 5b, MLX think-off, five nvfp4 models, `sync0341b_1_` against `cand034_1_` (the deployed 0.34.0 build)** —
render `preflight-runs/sync0341-render.md`. The chained run (`sync0341_`, 17:03–17:19) was invalid: MLX admission
refused 31b, 27b and 35b-a3b at every rung and 26b's escalation past 8192 — 64 refusals of the form "model requires
31.1–38.2 GiB … but only 28.6 GiB are available (after 16.4 GiB overhead)" — which was the foreign 38 GB, not the
image. `gate-sync0341-c.sh` re-ran the leg alone (`wait-gpu0-then-c.sh` waited for 60 GB free) on 2026-09-18
07:41–08:47 with GPU0 at 85 GB free: 5 suites, 0 errors, 0 OOMs, 0 not converged.

- Every scored cell equal within 0.003 IoU on all five models; contract matrices identical row for row; every
  multi-image and fine-text cell identical.
- One cell moved at n = 1: qwen3.6:35b-a3b name_bbox IoU 0.613 → 0.504 (in-band 4/5 both). Repeats
  `sync0341r_{1,2,3}_` (35b-a3b, full suite, 08:48–09:08, GPU0 otherwise idle; render
  `preflight-runs/sync0341-reps-render.md`) put the four reps on the new image at **0.504, 0.504, 0.613, 0.613** —
  the deployed build's value is one of the cell's two modes, and the fold reproduces it in 2 of 4. Fine text 7 px
  (1 → 2 in one rep) and the `bc`/`bcreasoning` contract flags flip between reps the same way. That is the known
  MLX think-off non-reproducibility across cold loads (memory note; ADR 0012 §4 does not cover MLX), not a build
  change: no MLX cell regressed.
- Throughput is not read: 12b ran first into the fresh `d9add9d1` PTX cache and paid the JIT (prefill 952 → 75
  tok/s, s/req 21.8 → 51.4); the later models move both ways (27b gen 34 → 58, 26b 65 → 35).
- The runner log carries six `custom GPU kernel backend disabled … backend=cuda reason="no source"` warnings
  (`gated_delta`, `gated_delta_states`, `depthwise_conv_silu`). That is upstream's own shape: those kernels ship a
  Metal source only, and `gpu_kernel.go` disables the CUDA path when `k.cuda.source == ""` and falls back to the
  composed ops; only `gated_delta_recurrence` has a CUDA source. Not a fold gap.

**Memory re-measure, from gate 5b's runner log (`preflight-runs/vsuite-sync0341b-runner.log`, 140 `memory`
lines):** `summarize_retained_memory.py` prints `held` — MLX's active memory after each request's scope ended and
the cache was cleared — and its step. The first real log exposed a parser bug: slog writes `msg=memory` unquoted,
the parser accepted only `msg="memory"` and read nothing; fixed in `runnerlog.py` with the real line pinned as a
test (133 tests). What the series says:

- gemma4 12b, 26b, 31b: `held` climbs by a fixed per-request amount (+0.95, +0.60, +2.4 GiB — the prefix trie's
  snapshot of each image request) and then plateaus: 12b flat at 15.1 GiB from request 9 to 27 (drift +0.07 GiB
  over 18 requests, ≤ 4 MiB per request), 26b flat at 24.3 GiB from request 14, 31b at 25 ± 0.4 GiB. A 144 MiB
  per-request leak — the 0.34.0 unit — would show as +0.14 GiB per request on those plateaus; it does not.
- qwen3.8:27b (the model the 0.34.0 leak was measured on): climbs to 36.8 GiB at request 17, then the trie pages
  out and `held` falls ~1 GiB per request back to 26.6 GiB by request 27. Memory that is released is tracked
  memory; the untracked residue cannot be read off this log because the trie's own size is not in it.
- **qwen3.6:35b-a3b: `held` grows +0.60 GiB per request for all 28 requests, 22.15 → 36.85 GiB, with no
  eviction.** This log cannot split that between the trie filling towards its cap (the run may simply have ended
  before the cap bound) and a leak on the one recurrent-state model in the set. The split needs the trie's
  accounting line (`prefix cache active_tokens … active_size … paged_out … snapshots`, `prefix_cache.go:793`,
  trace level, still present in the 0.34.1 runner): a probe with `OLLAMA_DEBUG=2`, same suite, 35b-a3b and 27b
  (`gate-sync0341-trace.sh`, prefix `sync0341t_`) ran after the repeats, and the summariser now subtracts the
  trie's active and paged-out bytes from `held` (one test).
- **Trace probe (09:09–09:25, `preflight-runs/vsuite-sync0341t-runner.log`): the growth is not the trie.**
  35b-a3b: `held − trie` grows +0.23 GiB per request in 24 of 27 steps, 21.9 → 27.2 GiB over 28 requests, while
  the trie grows 0.24 → 5.6 GiB. 27b: `held − trie` grows +0.6 to +1.1 GiB per request from request 3 to 20
  (16.8 → 32.8 GiB), then falls ~1.0 GiB per request for six requests with the trie flat at 7.9 GiB / 46
  snapshots, ending +10.6 GiB above request 1 — retention that outlives requests and is released in 1 GiB steps
  by something the log does not name.
- **Control (09:27–09:42, `sync0341n_`, same suite, `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`): flat.** 35b-a3b's
  `held − trie` is 21.91–22.04 GiB for all 28 requests (+0.13 GiB total); `held` is the weights plus the trie and
  nothing else. **The 0.34.0 finding stands on this build: drafting under a grammar retains memory outside the
  trie's books, +0.23 GiB per request on 35b-a3b, and the fork's knob (ADR 0033, D5) removes it entirely.**
- **Like for like against the deployed build (09:43–09:59, `cand034t_`, same trace suite on
  `maxusai/ollama:sync-0.34.0-main`):** the 0.34.0 runner logs its own `active` figure at the same point the
  0.34.1 runner logs `held` (MLX active memory after the teardown's cache clear), and both log the trie line, so
  resident memory and resident − trie compare directly (parser: `runnerlog.py`; logs:
  `preflight-runs/vsuite-{cand034t,sync0341t,sync0341n}-runner.log`).

  | model | build / arm | resident req 1 | req 14 | req 20 | req 28 | growth 1→28 | trie at 28 | resident − trie, 1→28 | max peak |
  |---|---|---|---|---|---|---|---|---|---|
  | qwen3.6:35b-a3b-nvfp4 | 0.34.0 deployed, drafting on | 22.1 | 25.9 | 27.6 | 29.2 | +7.1 | 5.6 | +1.8 | 39.2 |
  | qwen3.6:35b-a3b-nvfp4 | 0.34.1 fold, drafting on | 22.1 | 27.6 | 29.7 | 32.8 | +10.6 | 5.6 | **+5.3** | 44.8 |
  | qwen3.6:35b-a3b-nvfp4 | 0.34.1 fold + `ec3cc2307`, drafting on | 22.1 | 26.8 | 28.4 | 29.9 | +7.7 | 5.6 | **+2.4** | 40.0 |
  | qwen3.6:35b-a3b-nvfp4 | 0.34.1 fold, knob off | 22.1 | 25.0 | 26.2 | 27.6 | +5.5 | 5.6 | **+0.1** | 40.1 |
  | qwen3.8:27b-nvfp4 | 0.34.0 deployed, drafting on | 17.4 | 27.2 | 31.1 | 30.9 | +13.5 | 8.1 | +6.1 | 43.1 |
  | qwen3.8:27b-nvfp4 | 0.34.1 fold, drafting on | 17.4 | 34.8 | 41.3 | 35.4 | +18.0 | 8.1 | **+10.6** | 49.0 |

  (GiB; one rep each; the trie is the same size on every build, so the difference is all outside its books.)
  **Read:** drafting under a grammar retains about three times more on the 0.34.1 fold than on the deployed
  build — +5.3 against +1.8 GiB on 35b-a3b, +10.6 against +6.1 on 27b over 28 requests — and the fork's knob
  takes the fold to flat. Quality is unchanged either way (gate 5b). This is a memory regression of the MLX path
  for recurrent-state models under drafting + grammar, not a correctness one; the deploy has the knob
  (`OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`, kept as our option on 2026-09-12) to remove it, at the cost of drafting
  speed on grammar requests only.
- **Upstream's `ec3cc2307` on a Go-only swap (10:00–10:08, `sync0341k_`, image `sync-0.34.1-kvrel`, binary
  `0.34.0-dynres-18-g4cf8178` on the same payload): +2.4 GiB.** The pool-release cadence — firing only when the
  generated count lands exactly on a multiple of 256, which a speculative round steps over — is the part of the
  residual that is new in 0.34.1: with the hunk the fold retains +2.4 GiB against the deployed build's +1.8, without
  it +5.3. The drafting retention both builds share (+1.8 to +2.4) is not the pool; only the knob removes it.
  **Carried (D7): the hunk is cherry-picked onto this branch, re-rooted under `x/`, with the probe numbers in its
  message.** Quality was gated on the image without it; the hunk changes when the allocator's free pool is
  released, not what is computed, and the memory probe is its gate.

**Gate 5c, Qwen2.5-VL six cells, `q25vl0341_1_` against `q25vl_1_`** — render `preflight-runs/q25vl0341-render.md`:
every quality cell identical across all six models, contract matrices identical, 0 errors, 0 OOMs. Throughput
25–45 % lower under the same contention; not read.

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
- **D6 — `x/structured` deleted (Glenn, 2026-09-17).** The parity gate found 0 regressions in 108 verdicts; the
  package had no importers. Its findings are in ADR 0033's amendment and pinned by
  `x/mlxrunner/xgrammar/engine_behaviour_test.go`; the register moves it to retired.
- **D5 — the knob stays.** Upstream still drafts under a grammar; ADR 0033 is unchanged.

- **D7 — v0.34.2 is its own fold; `ec3cc2307` came forward on evidence (Glenn, 2026-09-18; probe below).**
  Upstream v0.34.2 (15 commits, 365 files, +3.7k/−70k) moves the MLX engine out of `x/`, re-lays out the models
  and bumps llama.cpp to b10969 — every fork path under `x/mlxrunner` re-homes, a structural fold, not widened into
  this one. Its 5-line "Release freed KV buffers during speculative decode" (`ec3cc2307`, the pool release firing on
  crossing a 256-token boundary instead of landing on one) is cherry-picked into this fold if the two memory probes
  below say it is the residual, and left to the 0.34.2 fold otherwise.
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

## The drafting retention, reproduced and bounded (2026-09-18, after the deploy)

Glenn asked for a deterministic reproduction before any upstream report. Method: one container at a time on GPU0 beside
production, `OLLAMA_DEBUG=2`, N identical or fresh-prefix requests, the runner's own `memory … held=` line and the trie's
trace line per request (`held − trie`, trie = active + paged-out), then instrumented Go-only swaps of the release image
(`sync-0.34.1-instr` … `instr6`, never deployed) that add per-bucket accounting, a live-array registry and counters.
Scripts `claude-scratch/leak-repro*.sh`, analysers `leak-repro-analyse.py` / `leak-repro-detail.py`, runner logs
`preflight-runs/leakrepro-*-runner.log`.

**Reproduction.** A request that carries an image, ends by `stop` (EOS or grammar completion) and runs under
speculative decoding retains memory across the request; nothing else does. The size ladder (the preflight's five
ladder images cycling, a bbox-per-object schema, `num_predict` 1500 — the ladder images hold nothing, so every answer
is 5–9 tokens and stops) grows on 14 of 14 steps on qwen3.6:35b-a3b (+4.5 GiB over 15) and on 27b; the single
stop-terminated request in an otherwise length-terminated run is the single request that grew (+422 MiB); 0 of 38
length-terminated image requests grew; text-only stop-terminated requests, identical or fresh-prefix, grew only by the
trie's own snapshots. The grammar is irrelevant (the no-format ladder leaks identically), so upstream's default
configuration is affected. `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0` (no drafting) is flat in every arm. `qwen3.5:0.8b-mlx`
cannot reproduce it: the package ships no MTP head and never drafts.

**Excluded, each by an instrumented run rather than by reading:** every Go-owned array (the registry shows zero
root-held arrays and an identical live set with and without drafting); unevaluated graphs (evaluating every live array
at teardown releases 0 B); the media fold's prefill captures (disabling them entirely: identical growth); the
drafter's caches and media rows and the target cache slots (their buffers are flat by size); speculation's per-round
snapshots (created = closed, 20,970 = 20,970); MLX-C handle leaks (every vector/closure site has its free); MLX's
compile cache (`MLX_DISABLE_COMPILE=1`: unchanged); the CUDA graph cache size (24 requests at 400/20/50: +7.9 /
+5.8 / +7.9 GiB — within run-to-run variance; an 8-request run that looked flat at 20 was n = 1 and wrong);
synchronous evaluation of the round's drafts (two runs, unchanged). The pool-release cadence was the part
`ec3cc2307` removed. What remains is counted by MLX's allocator but referenced from the C++ side of the boundary.

**Two components, measured on the same requests.** Sampling the runner's device memory (nvidia-smi) beside MLX's
counter over 12 stop-terminated image requests: device +6.8 GiB, MLX active +3.2 GiB, of which the trie +1.7 and the
unowned MLX-tracked part +1.2 — so more than half of the growth sits **outside MLX's accounting** (device − active
2.2 → 5.8 GiB): CUDA-internal allocations that neither `held` nor the admission headroom can see. Bounding the graph
cache shrank that outside part (+2.2 instead of +3.6) without touching the MLX-tracked part, which is what an LRU of
instantiated graph execs would do; it is a share, not the mechanism.

**The knob, measured on device memory (two runs, 12 stop-terminated image requests each, no drafting):** MLX's counter
is flat outside the trie (−0.12 GiB both runs) — component A is gone — but **device memory still grows +6.8 and +6.6
GiB**, more than with drafting. So the CUDA-internal component B is not drafting's: it follows **shape variety** (a
fresh prefix per request is a new prefill shape, the ladder a new image shape), i.e. instantiated CUDA graph execs and
JIT'd kernels per distinct shape, bounded by MLX's 400-entry graph cache at whatever a big prefill graph costs. The
length-terminated control (fixed scene image, 1,500-token answers, drafting on) grows +1.0 GiB of device memory over
12 and +0.3 of MLX-tracked memory — its two requests that happened to stop early. Two components, two owners:

| component | condition | seen by | size (35b-a3b) | what removes it |
|---|---|---|---|---|
| **A** MLX-tracked, owned by nothing on the Go side | image + stop-terminated answer + speculation, grammar or not | `held`, admission | ~0.1–0.3 GiB per such request, unbounded | no drafting (`OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0` covers grammar requests only); no code fix known — the holder is on MLX's side |
| **B** CUDA-internal, outside MLX's allocator | every new input shape (prompt length, image size), drafting or not | nvidia-smi only | +0.5 GiB per new shape early, bounded by the graph cache | a smaller `MLX_CUDA_GRAPH_CACHE_SIZE` (prefill latency cost), or pricing it into the admission headroom (ADR 0034) |

**B's plateau (60 no-drafting requests, fresh prefix each, 19:25–19:33):** the CUDA-internal part peaks early and
comes back down — device − MLX-active +6.6 GiB at request 10, +5.2 at 20, +3.8 at 60 (cache 400); +5.8 / +4.3 /
+2.7 at cache 50 — so it is a transient of the first distinct shapes plus a bounded steady state, not a leak; the
bound at 50 saves ~1.1 GiB with no visible latency cost at steady state (1.5–1.8 s per request either way). Over the
same 60 requests the trie fills to its 8 GiB `maxPagedOutBytes` budget (8.11 GiB at request 50 and flat after, exactly
ollama#17924's plateau) and MLX-tracked memory outside it stays at −0.2 GiB. Steady state for 35b-a3b without
drafting: **~34 GiB of device memory** = 22 weights + 8 trie + ~4 CUDA-internal, against an admission that prices the
weights and the rung's KV. **A is the only unbounded growth**: +7.9 GiB over 24 stop-terminated image requests with
drafting, linear, no plateau in sight.

**Context upstream.** ollama#17924 (closed by its reporter as the trie filling to its 8 GiB `maxPagedOutBytes`)
measured 0.147 GiB per request on this model family; ollama#17875 and #18131 report growth past that budget on Metal
under agent workloads (short, stop-terminated answers with long contexts) and were closed as trie behaviour. This
investigation supplies the non-trie component those reports were missing, with its condition.

## The gemma4 image resize algorithm (Glenn's question, 2026-09-18)

Glenn asked whether `hparams.image_resize_algo = RESIZE_ALGO_BILINEAR → RESIZE_ALGO_BICUBIC` in gemma4's projector case
came up as a regression candidate. It did not, and for this fold that was right: `clip.cpp` has `RESIZE_ALGO_BICUBIC`
for `PROJECTOR_TYPE_GEMMA4V/GEMMA4UV` at **both** b10760 and b10864, so the switch is not in this fold's delta.
Where it did enter the fork: llama.cpp `56db501e7` "mtmd: use pillow-accurate algo, correct resize_algo for all models
(#27594)", between b10488 and b10630 — the **0.33.1 fold** — and no fork record flagged it then. That is the gap.

What the campaigns on record say about it, for the four gemma4 GGUF models: the 0.33.2 (b10630, bicubic) and 0.33.3
(b10760, bicubic) campaigns are identical cell for cell; against the 0.33.0-line `sync15_` run (b10488, bilinear)
the only visible move is e4b's 12 px fine-text tier 0 → 3 (n = 1 each), in the direction bicubic downscaling would
give. gemma4:e2b's zeros (fine text 0/0/0/0/0, scene IoU 0.061, invoice 1/5) exist on every build from b10488 to
b10760, bilinear and bicubic alike, and recover only at b10864 — where the one gemma4 commit is `163a40796` "model,
mtmd: fix gemma4 vision handling (#28335)", the same commit that moved the default limits to (70, 1120). So the
e2b/e4b recovery this fold measured is attributable to #28335, not to the resize algorithm. **Veto closed:** the
four gemma4 cells re-run on the deployed 0.34.0 image on 2026-09-18 (`ggml034main_1_`, b10760) reproduce the baseline
cell for cell — e2b's zeros, e4b's 0.354 / 0.000, 26b-a4b's `bcreasoning`/`bcpinned` both ✅ — so GGUF think-off is
deterministic on both payloads and every gemma4 move in this fold is #28335's. Its mechanism for e2b/e4b: the commit
carves E2B/E4B out of non-causal image decoding (they decode images causally now), which is what unbroke them.

**The same commit is a regression upstream — llama.cpp #28954 (2026-09-15):** raising gemma4's cap to 1120 image
tokens exceeds llama-server's default `n_ubatch` of 512, and a non-causal image chunk larger than one ubatch trips
`GGML_ASSERT(cparams.causal_attn || cparams.n_ubatch >= n_tokens_all)` in `llama_context::decode`, which aborts the
whole llama-server — every image above ~1.2 Mpx on 12b/26b/31b. First bad commit bisected to `163a40796`. **The fork
does not crash**: our launcher passes `-b N -ub N` (equal, `appendBatchArgs`), and `mtmd_helper_decode_image_chunk`
splits a chunk into `n_batch`-sized decodes, so every decode fits its ubatch. What that split costs instead: with
the fork's automatic generation batch of 1024 (ctx > 4096, `automaticGenerationBatch`), a top-rung gemma4 image
(1117–1121 tokens) is decoded as two non-causal sub-batches, and the first 1024 image tokens never attend to the last
~95 — one-way bidirectional attention at the top rung, for as long as the fork has served 1120 (ADR 0008), on both
payloads. Not a fold regression; a latent inconsistency the gates have measured consistently. The clean fork fix is
a generation batch ≥ the image-token ceiling for gemma4 vision runners (2048), at the compute-buffer cost of the larger
ubatch; the measurement that decides it is the top-rung bbox cells at `num_batch` 1024 against 2048. Register entry
added; the upstream ollama launcher (v0.34.2 `llm/server.go`) is checked below for its own exposure.

**Measured (2026-09-18, 21:24, `claude-scratch/batch-ab-0341.sh`, render `preflight-runs/batchab-0341-render-thinkfalse.md`):**
the deployed 0.34.1 image, GPU0 beside production, the two gemma4 GGUF models that decode images non-causally, think-off
at the 8192 rung, two repeats per arm. `batch1024_` is the scheduler's automatic batch at that rung (the runner log shows
every image chunk, 1064–1100 tokens on these fixtures, decoded as `1/2 n_tokens_batch = 1024` plus a 40–76-token
`2/2`); `batch2048_` sends `num_batch: 2048` per request (`NUM_BATCH` in `client.py`, branch `vsuite/num-batch-env`),
and the same chunks decode as `1/1`. Both arms are deterministic: every scored cell is identical between repeats.

| test | metric | 31b, batch 1024 | 31b, batch 2048 | 26b-a4b, batch 1024 | 26b-a4b, batch 2048 |
|---|---|---|---|---|---|
| scene | bbox IoU | 0.966 | 0.963 | 0.978 | 0.977 |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.709 | 0.709 | 0.753 | 0.753 |
| fine text | 22/16/12/9/7 px | 4/4/4/**4**/3 | 4/4/4/**3**/3 | 4/4/4/3/3 | 4/4/4/3/3 |
| multi (3 img) / anchored | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 | ✅ ✅ ✅ 5/5 | ✅ ✅ ✅ 5/5 | ✅ ✅ ✅ 5/5 |
| throughput | prefill tok/s | 474 / 508 | 753 / 721 | 1250 / 1369 | 2434 / 2518 |
| latency | s/req (unique image) | 12.7 / 12.6 | 11.5 / 11.6 | 4.5 / 4.2 | 3.7 / 3.7 |

(T2 rows from `summarize_head_to_head.py`; throughput and latency give both repeats.) So the split costs nothing the
scored cells can see: IoU within 0.003, every contract identical. Decoding the chunk in one piece is what the batch
buys — prefill 1.5× on 31b and 1.9× on 26b-a4b, s/req −9 % and −18 % — and it moves one cell: 31b's 9 px fine-text
tier, 4 with the split and 3 without, deterministic 2/2 on each side. That is the same shape as the Metal finding in
#312 (the more correct encoder path scores one 9 px tier lower on 31b), and the same caveat applies: a single
knife-edge tier is not a quality verdict. Glenn's decision, the same evening: raise it to the 1120 ceiling — ADR 0036, #320 (merged 2026-09-18): a gemma4
vision runner starts from the smallest batch rung at or above its resolved image ceiling and steps down only when
that does not fit; the register row reads "fixed in the fork".

The same container answered #313's ask from the ROCm host (`run_budget_sweep.sh`, `BUDGETS="280 560 1120"`, min == max,
gemma4:31b, num_ctx 16384 so the 1120 rung decodes past n_ubatch = 1024): scene IoU 0.925 / 0.936 / 0.967 and
name_bbox 0.691 / 0.714 / 0.709 — no cliff where gfx1151 collapses to 0.000 — with `prompt_eval_count` 848 / 614,
1111 / 887, 1684 / 1447 (scene / document), digit for digit the ROCm host's b10864 figures. The budget-fill geometry
is identical across the two hosts; the gfx1151 defect needs the HIP `integrated` path, not the batch geometry.

The fork's `004` fill resizes through `hparams.image_resize_algo`, so it has followed bicubic since 0.33.1; the
preflight's `token_ladder` (5/5 geometries) and `pinned_image_token_budget` (560 → 529, ceiling 1120) pass on it.

## Retirement candidates (Glenn, 2026-09-17)

`x/structured` was tested against upstream's engine (108 verdicts, 0 regressions) and deleted in this fold on
Glenn's word (`070580c5e`, `b1db10efc`). That gate, and every other item the fork carries with its retiring condition and gate, are in
[`docs/maxusai/retirement-register.md`](../retirement-register.md), reviewed at each fold. Two rows moved in this
one: upstream 0.34.1 has its own transition-based `format` deferral with pass-one metrics (ours is now a superset,
README row corrected), and `extendChunk` from ADR 0014 turns out to be upstream's already.

## Release and deploy (2026-09-18)

PR #302 merged by Glenn (`8a7ba9498`, 11:56); tag `v0.34.1-dynres` on it; release published with the fold's matrix and
the memory table. Release image `maxusai/ollama:sync-0.34.1-main` (`e224595df7e0`, stamp `0.34.1-dynres-0-g8a7ba94`)
built from a worktree at the tag on the `bigdisk` builder — every native stage cached from the candidate build, 2.5 min.

- **Attempt 1 was stopped by the root-disk watchdog** at the image load: the daemon writes the whole image tar to
  root before it deduplicates layers, ~6 GB transient, which took root from 13 GB to 6.7 GB free, under the 8 GB
  floor that protects production. Nothing in the fork's own images frees enough (their layers are shared; each old
  tag is ≤ 80 MB unique). Glenn removed the rotated `syslog.1` (5.2 GB); attempt 2 ran at 18 GB free. Rule for the
  next fold: **≥ 16 GB free on root before a `--load`.**
- **Deploy gate** (`gate-v0341-main.sh`, 13:31–13:37): all 57 native payload files hash-identical to the gated
  candidate `sync-0.34.1`; the Go binary differs as expected (`x/structured` gone, `ec3cc2307`, a comment).
  Preflight `cuda-dynres-903` from the main checkout on a canary of the release image: **PASS 21 / SKIP 7**
  (`preflight-runs/full-v0341-main.{log,json}`).
- **Deployed 14:01:00** by `deploy-0341.sh` (mirrors production by `docker inspect`; Glenn's word 13:5x, with the
  knob): container `ollama-0.34.1-dynres-0-g8a7ba94` on `0.0.0.0:11497`, image `sync-0.34.1-main`, env
  `OLLAMA_HOST` + **`OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`** (D5: the retention both builds share under drafting +
  grammar goes to flat; costs drafting speed on grammar requests only), 54 tags before and after, 10 s of no service.
  `ollama-0.34.0-dynres-0-gcf2ad41` is stopped and kept: rollback is
  `docker rm -f ollama-0.34.1-dynres-0-g8a7ba94 && docker start ollama-0.34.0-dynres-0-gcf2ad41`. The teacher-v3
  session confirmed nothing of theirs was on `:11497`.
- **Post-deploy preflight against `:11497`** (14:01–14:11, `preflight-runs/full-0341-deployed.{log,json}`): **PASS 21 /
  SKIP 7**; version, both payload pins, budgets, think + format and the poison probe all green on production itself,
  and the run paid the cold kernel compile so the first real request does not. The README's matrix is regenerated from
  this run, stamped `0.34.1-dynres-0-g8a7ba94`.
- **ADR 0036 deployed 22:30:18** (Glenn's word the same evening, after #320 merged) by `deploy-adr0036.sh`, the same
  mirror-by-`docker inspect` script: container `ollama-0.34.1-dynres-16-g16649e8`, image
  `maxusai/ollama:0.34.1-dynres-16-g16649e8` = the release image with a Go-only binary swap from `main` `16649e8`
  (`publish-go` on the `bigdisk` builder, seconds; `git diff 8a7ba949...16649e8` over every native input — `x/mlxrunner/mlx`,
  the MLX and llama.cpp pins, `cmake/`, `llama/`, `ml/`, the Dockerfile — is empty apart from the HIP-only compat 906
  patch, which the CUDA payload compiles to the same `integrated = false` either way), same env (`OLLAMA_HOST` +
  `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`), 54 tags before and after, 11 s of no service. `ollama-0.34.1-dynres-0-g8a7ba94`
  is stopped and kept: rollback is
  `docker rm -f ollama-0.34.1-dynres-16-g16649e8 && docker start ollama-0.34.1-dynres-0-g8a7ba94`.
  Verified on production itself: the first gemma4:31b-it-q4_K_M request launched `llama-server … --image-max-tokens 1120
  … -b 2048 -ub 2048` and logged `decoding image batch 1/1, n_tokens_batch = 1100` (scene bbox IoU 0.963, 6 labels,
  prompt_eval 1684, `preflight-runs/prod-adr0036-g31b.log`; the request paid the cold kernel compile, 52 s for that one
  image decode, as every first request after a deploy does); the MLX runner (gemma4:12b-nvfp4, `keep_alive` 0) answered `OK` — the first request paid the runner's cold JIT (8 m 59 s;
  a 600 s client cap on the first attempt tripped before it, logged as `500 10m0s`), the second loaded in 10.5 s. Both
  models unloaded; `/api/ps` empty afterwards. The README's matrix keeps the tag's full
  preflight stamp: the swap changes no native input, so the run is not repeated.

## The MLX pin move and the vision encoder (#312, 2026-09-18)

The Metal session found that the `gemma4:31b-nvfp4` and `26b-nvfp4` image embeddings (`TestVisionGoldenParity`, against
the vendored mlx-vlm reference) move across `2b95b4a5 → 8a7ba949` on Apple Silicon while `12b-nvfp4` is bit-identical,
and asked this host for the same measurement on MLX-CUDA, to split "Metal kernels" from "shared Go/MLX-C code" (#312).
Measured here at both ends: the Go side at each commit in a `golang:1.26` container, dlopen'ing the payload extracted
from the fork's own release image for that commit (`sync-0.33.2` = `0.33.2-dynres-5-g2b95b4a`, MLX `c793734e`;
`sync-0.34.1-main`, MLX `d9add9d1`), goldens byte-identical at both commits, two runs per cell, every repeat to the digit:

| model | `2b95b4a5` mean / std / norm_mean / max Δ | `8a7ba949` | Metal, from #312 |
|---|---|---|---|
| 12b | −0.02562 / 2.4605 / 151.309 / 0.0625 | identical in every digit | identical (151.310 / 0.0625) |
| 26b | −0.00024 / 1.6290 / 86.343 / 0.0469 | identical in every digit | 86.346 / 0.2266 → 86.350 / 0.0508 |
| 31b | −0.00621 / 1.3243 / 96.999 / 0.0898 | −0.00614 / 1.3251 / 97.056 / 0.1094 | 96.749 / 0.1406 → 96.998 / 0.1094 |

Two causes, one per backend:

- **Metal-only, and a fix.** ml-explore/mlx#3912 (`dfe17baf`, merged 2026-09-11, inside the pin range, Metal kernels
  only): `fp_qmm_t` ran its K loop past `K_eff` when **K % 32 == 16**, reading 16 columns of the next row's packed weights
  and scales into the accumulator, for every matrix-sized nvfp4 `quantized_matmul`. The gemma4 vision tower's
  `mlp.down_proj` is nvfp4 `[1152 × 4304]`, K = 134 × 32 + 16, in 26b and 31b (27 layers each); 12b has no tower, and
  its two quantized vision weights have K = 3840 and 6912. So the 0.33.2 Metal encoder for 26b/31b was corrupted (the
  Metal session then reproduced it at kernel level: 79 % of outputs off by > 1.0 at K = 4304, M = 256; clean at
  K = 4288) and 0.34 moved *toward* the reference. The CUDA kernel (`qmm_sm80_tn_…_g16_nvfp4`) never had the defect:
  26b reads 0.0469 at both ends, the number the fixed Metal kernel (0.0508) now agrees with. **No campaign on this host
  is affected** — mlx-cuda never ran the Metal kernel, so the nvfp4 campaigns here
  (`vision-campaign-2026-08-28-mlx0330-nvfp4.md`, `…-08-31-mlx0332-nvfp4.md`) stand; Metal-side 26b/31b vision numbers
  from before `8a7ba949` were measured through the corrupted kernel.
- **Shared Go code, small.** The fold's `ToMLXGlobalScale` (`x/mlxrunner/model/quant.go`) holds every nvfp4 global
  scale as `f32(m × 2688)` (MLX's representation, 2688 = 448 × 6) and `scaleAndCast` (`x/mlxrunner/mlx/ops_extra.go`)
  applies `f32(f32(m × 2688) / 2688)`, which is one f32 ulp off `m` for 17 of 31b's 191 vision global scales and exact
  for both of 12b's; 26b's tower has none. Control: `8a7ba949` with a two-line revert of only that arithmetic
  (`claude-scratch/wt-gsexact`), same 0.34.1 payload, reproduces `2b95b4a5`'s 31b line to every digit. Inside the
  test's bounds (norm_mean +0.06 %); not a regression. Whether to keep the raw multiplier on the wrapper-applied paths
  (`QuantizedMatmul`, `Dequantize`) and reserve `× 2688` for the paths that hand the scale to MLX (`GatherQMM`, `QQMM`)
  is Glenn's call; the reference uses the raw `m`.

The gap is the same class as the resize algorithm: the MLX pin was folded as an opaque range. **Rule for the next
fold:** list the range's commits (`gh api repos/ml-explore/mlx/compare/<old>...<new>`, then each commit's files) and
check every kernel fix's selection condition against the served models' quantized tensor dims, read from the store's
safetensors headers (nvfp4: `weight` U32 columns × 8 = K, `.scale` U8 columns × 16 = K). A fix selected by shape moves
one backend for a subset of models and reads exactly like a shared-code drift.

Running `x/mlxrunner` tests against the CUDA payload outside the image (`claude-scratch/golden312.sh`): `golang:1.26.0`
with `-u 1000:1000` and `--gpus '"device=0"'`; the payload from `docker cp <image>:/usr/lib/ollama/mlx_cuda_v13`
symlinked at `<wt>/build/lib/ollama/mlx_cuda_v13` **plus** `<wt>/build/lib/ollama/include → mlx_cuda_v13/include`
(MLX finds CCCL relative to the library's parent, not `CUDA_PATH`; without it the first JIT fails on
`cuda/std/tuple`); `LD_LIBRARY_PATH`, `CUDA_PATH` and `CUDA_HOME` at the payload dir; one `MLX_PTX_CACHE_DIR` per MLX
version; `OLLAMA_MODELS` at the store; `OLLAMA_VISION_E2E=1`. An old Go commit loads the old image's payload with no
rebuild, which is #307's point. 12b ≈ 130 s cold, 31b ≈ 25 s warm.

## Not in this fold

- The Metal half: MLX and MLX-C moved, so the Metal payload changes too; held by Glenn.
- `v0.34.2-rc1` (8 commits further) moves the MLX engine out of `x/` and re-lays out the models; every fork path
  under `x/mlxrunner` re-homes. That is a separate, structural fold.
- PR #301 (Glenn's whitespace bound, renumbered to ADR 0035, green): merging it before this branch lands means a
  small merge of `client.go` here; after, it rebases onto the resolved file. Either is fine.
