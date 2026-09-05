# TASK: price the context rung in MLX admission, and pre-size the KV

**Opened:** 2026-09-05. **Status:** CODE LANDED 2026-09-05 on branch
`feat/mlx-admission-price-the-rung` (fix 1 only); **GPU verification outstanding**
and fix 2 (pre-sizing) not started. See
[ADR 0034](../adr/0034-mlx-admission-prices-the-context-rung.md) for the decision
and [What landed](#what-landed-2026-09-05) below for the state of each criterion.
**Report:**
[`mlx-admission-prices-weights-only.md`](../mlx-admission-prices-weights-only.md)
(merged as [#210](https://github.com/MaxusAI/ollama/pull/210); issues are disabled on
this repo, so that note is the diagnosis and this is the work item).

## Why now

#210 was written after `gemma4:31b-coding-mtp-bf16` produced 13 OOM aborts in one
vision-suite cell. It was read as a one-model problem and left open. The v0.33.3 fold's
Gate 5 spot-check (2026-09-04) shows it is not:

- Four arms aborted with `cudaMallocAsync … out of memory` on `gemma4:26b-nvfp4` and
  `gemma4:31b-nvfp4` — **two of them small single-image requests**, which no
  request-size argument explains.
- Per-request peak memory that run: **p50 24.4 GiB, p90 33.1 GiB, max 36.3 GiB**, against
  weights of 17–24 GB. The unpriced remainder is the whole margin.
- The same four arms re-ran **12/12 clean** with `OLLAMA_MLX_MEMORY_LIMIT=40 GiB`, and the
  campaign's neighbour on the shared GPU (the teacher-v3 fetch loop on `:11497`) was
  moving free memory throughout.

So: **contention is the trigger, an unpriced KV is why there is no margin to absorb it.**
Both halves are needed — that is the standing explanation for this abort class, and it is
why campaign and canary containers now need `OLLAMA_MLX_MEMORY_LIMIT` by hand (#272).

## What is true in the code today (verified 2026-09-05 on `main` @ 2d2812eb)

- `req.opts.NumCtx` **does** reach the MLX client — `server/sched.go:609` calls
  `mlxrunner.NewClient(modelName, softContextLength)`. It is used in exactly one place,
  `client.go:259-260`, to cap the *served context length*. **It never enters the memory
  arithmetic.**
- Admission prices `c.memory.Store(TotalTensorSize())` (`client.go:65`) — weights only —
  against `gpus[0].FreeMemory`, at `client.go:~330` and `:~359`.
- `cache.NewKVCache()` sets `step: 256` and grows by `Concatenate`; nothing preallocates
  from `num_ctx` (`git grep -iE 'numCtx|capacity' -- x/mlxrunner/cache/` is empty).
- Consequence, and the part to internalise: **`8192 → 16384 → 32768 → 65536` admit
  identically on MLX.** The ladder rung is invisible to admission, so the vision suite's
  own escalation cannot be refused early — it aborts mid-prefill instead.
- The v0.33.3 fold changed **none** of this (`git diff b54d4d0d origin/main --
  x/mlxrunner/client.go` over memory/budget lines is empty). Nothing regressed; the
  capability was never built.

## The two fixes #210 proposes

1. **Price the rung**: compare `TotalTensorSize() + kvBytes(num_ctx) + activation
   headroom` instead of weights alone. Converts a mid-prefill abort into the actionable
   refusal `client.go:360` already knows how to emit.
2. **Pre-size the KV from `num_ctx`** — allocate once at session start rather than
   concatenating upward. The precondition for (1) being accurate rather than approximate,
   and it removes the growth moment that transiently holds both buffers.

## Design notes for whoever takes it

- **The geometry is reachable before load.** `manifest.LoadManifest` already opens the
  model; `ModelManifest.ReadConfig` / `GetConfigLayer` expose `config.json`, and the model
  packages parse exactly what a KV estimate needs (`gemma4.go:44-49`:
  `num_hidden_layers`, `num_key_value_heads`, `head_dim`). No weight load is required to
  price a rung.
- **A naive `layers × kv_heads × head_dim × 2 × bytes × num_ctx` over-prices gemma4 badly.**
  Sliding layers use `cache.NewRotatingKVCache(m.SlidingWindow)` and are bounded by the
  window, not by `num_ctx`; only the global layers grow with the rung
  (`gemma4.go:1126-1131`, same pattern in `dflash` and `cohere2_moe`). The estimate has to
  follow the same sliding/global split the model itself uses, or a 31b at 65536 is refused
  on a card that would have served it. Gemma 4 also carries a separate `global_head_dim`.
- **Recurrent architectures** (`cache/recurrent.go`) do not scale with `num_ctx` at all —
  a per-cache-kind estimator, not one formula.
- **Over-refusal is the failure mode to avoid.** The current behaviour wrongly admits;
  a careless fix wrongly refuses, which is worse for a serving host. Bias the estimate low
  and keep `OLLAMA_MLX_MEMORY_LIMIT` as the operator's hard ceiling
  (`budgetWithOverride` already folds it in as `min(derived, requested)`).
- **`FreeMemory` is one optimistic sample** taken per reload; pricing the rung does not fix
  a neighbour that moves afterwards. The two are complementary: the estimate buys margin,
  the operator cap bounds the pool.

## Acceptance criteria

1. ☑/☐ A KV estimator that follows each model's own cache kinds (rotating vs full vs
   recurrent), unit-tested against at least gemma4 (sliding + global), a non-sliding dense
   model, and a recurrent one; **the numbers checked against a measured load, not
   asserted.** Estimator landed (`x/mlxrunner/kvsize`) and unit-tested against the real
   gemma4:26b, qwen3.6:35b-a3b and qwen3.8:27b configs plus synthetic nemotron_h and
   llama ones. **The half that is not done is the one that matters most: nothing has been
   compared to a measured load yet.**
2. ☑ Admission compares weights + KV(rung) + headroom, and the four ladder rungs
   admit differently — `TestLadderRungsAdmitDifferently` fails on the pre-change
   code (`num_ctx 16384 needs 42949672960, not more than 42949672960 at 8192`).
3. ☑ A refusal in `client.go`'s existing shape, naming weights / KV / rung / headroom /
   available and ending with the `runner.go:548` phrase `lower num_ctx or free VRAM on
   the device`.
4. ☐ KV preallocated from `num_ctx` at session start (#210 fix 2). **Not started.**
   Until it lands the estimate describes what the cache *will* hold, and the
   `Concatenate` growth moment still transiently holds two buffers.
5. ☐ No new over-refusal: the vision-suite cells that pass today still admit at their
   converged rungs on the CUDA host, verified by a think-off campaign run. **Not run.**
6. ☑ (code) / ☐ (verified) `OLLAMA_MLX_MEMORY_LIMIT` still wins as the ceiling and keeps
   its own separate plain error — `TestOperatorCapCoversTheContextRung`. The `801`/preflight
   surface is untouched by this diff but has not been re-run.

## What landed (2026-09-05)

Branch `feat/mlx-admission-price-the-rung`, two commits, no push.

- **`x/mlxrunner/kvsize`** — pure Go, no cgo, no `x/models` or `x/mlxrunner/mlx`
  import. `Model(config, draft, numCtx) Estimate` dispatches on
  `architectures[0]` (falling back to `model_type`) and mirrors each model
  package's `NewCaches`. `Estimate.Known == false` for an architecture with no
  rule, and the caller must then fall back rather than refuse.
- **`Client.admit`** (extracted from `Load`, which now calls it) prices
  `weights + KV(num_ctx) + headroom`.
- **`NewClient(modelName, numCtx, numCtxAuto)`**; `server/sched.go:609` passes
  `req.numCtxAuto`. `softContextLength` is now `atomic.Int64` because admission
  may write it while `Ping` reads it.

### The auto-clamp decision, and why

An **explicit** rung that does not fit is **refused**; an **automatic** one is
**clamped** — halved until it fits, floor 2048, with `softContextLength` updated
so `reportedContextLength` serves the fitted window.

The asymmetry is not a hedge. An explicit `num_ctx` is a request, and serving a
smaller window silently would be a lie the caller cannot see. An automatic
`num_ctx` is the VRAM-tier default — **262144 on this host** — that nobody chose;
pricing it and refusing would refuse nearly every model on the card, which is
exactly the over-refusal this work is supposed to avoid. An auto rung therefore
**never errors** unless the weights alone do not fit: if even 2048 does not fit,
admission falls back to weights-only and logs a warning.

### The headroom placeholder

`headroom = max(512 MiB, 5% of (weights + KV))`, labelled as a placeholder in the
code. It stands in for prefill activations, the vision tower and the
cache-growth double-hold, and it is **not** a model of any of them. Biased low
deliberately.

### What the estimator says now (so the GPU phase has something to falsify)

Estimated cache bytes, from the fixtures in `x/mlxrunner/kvsize/testdata`:

| model | 8192 | 16384 | 32768 | 65536 | 262144 |
|---|---|---|---|---|---|
| `gemma4:26b` (5 full + 25 sliding@1024) | 360 MiB | 520 MiB | 840 MiB | 1.45 GiB | 5.20 GiB |
| `qwen3.6:35b-a3b` (10 full + 30 recurrent) | 221 MiB | 381 MiB | 701 MiB | 1.31 GiB | 5.06 GiB |
| `qwen3.8:27b` (16 full + 48 recurrent) | 659 MiB | 1.14 GiB | 2.14 GiB | 4.14 GiB | 16.14 GiB |

**Read this before calibrating:** for gemma4 the KV is a few hundred MiB against
weights of 17–24 GB and observed peaks of 24–36 GiB. **The KV is the smaller half
of the unpriced remainder.** Pricing the rung makes the ladder visible and turns
an abort into a refusal; it does not explain the peaks. The headroom does, and it
is a guess.

### What the GPU phase must measure

For each of `gemma4:26b-nvfp4`, `gemma4:31b-nvfp4`, `qwen3.6:35b-a3b` and
`qwen3.8:27b`, at each rung 8192 / 16384 / 32768 / 65536:

1. **Estimate vs actual.** The admission line
   `MLX admission priced the context rung` carries `weights`, `kv`, `headroom`,
   `need`, `num_ctx`, `budget` and the per-kind layer counts. Compare `need`
   against the runner's `pipeline.go` `peak memory` line for the same cell —
   **text-only first, then the multi-image cell**, because the 3.8k-token image
   prefill is what moved the peak from ~67 to ~77 GiB in the original diagnosis.
   Record `peak − (weights + kv)` per rung: that difference is what the headroom
   has to cover, and whether it grows with the rung or with the image count is
   the open question.
2. **Whether the layer counts are right.** `layers_full` / `layers_sliding` /
   `layers_recurrent` in that log line must match what the model actually built.
   A mismatch means `kvsize` has drifted from a `NewCaches`.
3. **No new over-refusal** (criterion 5): the vision-suite cells that pass today
   must still admit at their converged rungs. A refusal that names a rung the
   card used to serve is a calibration bug, not a success.
4. **The sliding transient.** `RotatingKVCache.concat` (the batched prefill path)
   trims to `window-1` and concatenates the whole chunk, so a 2048-token prefill
   chunk holds up to `window-1+2048` slots — 3x the window for gemma4's 1024.
   The estimator prices the **steady-state** `min(roundUp(num_ctx,256), window)`
   and leaves that transient to the headroom. Measure whether it needs its own
   term.
5. **The auto clamp on a real load**: start a server with no explicit `num_ctx`,
   confirm the clamp line `MLX context clamped to fit VRAM` appears with a
   sensible rung, and that `/api/show` and the served context report the clamped
   value rather than 262144.

## Not in scope

The **3 illegal-memory-access aborts** #210 records are a different failure class, left
unexplained there and still unexplained. Do not fold them into this work; if the KV fix
makes them disappear, that is evidence, not a claim to assert up front.
