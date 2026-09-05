# TASK: price the context rung in MLX admission, and pre-size the KV

**Opened:** 2026-09-05. **Status:** OPEN, unassigned. **Report:**
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

1. ☐ A KV estimator that follows each model's own cache kinds (rotating vs full vs
   recurrent), unit-tested against at least gemma4 (sliding + global), a non-sliding dense
   model, and a recurrent one; the numbers checked against a measured load, not asserted.
2. ☐ Admission compares weights + KV(rung) + headroom, and the four ladder rungs
   **admit differently** — a test that fails on today's code.
3. ☐ A refusal at `client.go:360`'s shape naming the rung and what to lower, not an abort
   mid-prefill.
4. ☐ Optional but preferred: KV preallocated from `num_ctx` at session start (#210 fix 2),
   with the growth-moment double-hold gone.
5. ☐ No new over-refusal: the vision-suite cells that pass today still admit at their
   converged rungs on the CUDA host, verified by a think-off campaign run.
6. ☐ The `801`/preflight surface unchanged; `OLLAMA_MLX_MEMORY_LIMIT` still wins as the
   ceiling.

## Not in scope

The **3 illegal-memory-access aborts** #210 records are a different failure class, left
unexplained there and still unexplained. Do not fold them into this work; if the KV fix
makes them disappear, that is evidence, not a claim to assert up front.
