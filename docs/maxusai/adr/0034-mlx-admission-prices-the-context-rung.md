# ADR 0034: MLX admission prices the context rung

- **Status:** accepted (2026-09-05), code landed, **not yet GPU-verified**
- **Date:** 2026-09-05
- **Deciders:** Glenn; work item
  [`tasks/mlx-admission-price-the-rung.md`](../tasks/mlx-admission-price-the-rung.md),
  diagnosis [`mlx-admission-prices-weights-only.md`](../mlx-admission-prices-weights-only.md)

## Context

The MLX client priced a load as `TotalTensorSize()` — weights only — against
`gpus[0].FreeMemory`. `num_ctx` reached the client but was used in exactly one
place, to cap the *served* context length; it never entered the memory
arithmetic. So **8192 → 16384 → 32768 → 65536 admitted identically**, and a rung
that could not fit was accepted and then killed mid-prefill by
`cudaMallocAsync`. One campaign attempt produced 13 OOM aborts and 13 reloads of
a 63.5 GB model without completing a single cell.

The llama.cpp path prices `n_ctx` before the runner starts. This one did not.

## Decision

Admission compares `weights + KV(num_ctx) + headroom`.

**1. The KV estimate is a separate pure-Go package**, `x/mlxrunner/kvsize`. It
reads the manifest's `config.json` (and `draft/config.json`) and dispatches on
`architectures[0]`, mirroring each model package's `NewCaches`. It imports
neither `x/mlxrunner/mlx` nor `x/models/...`, because either would drag MLX into
the server binary. The cost of that separation is a **copy**: when a model
changes its cache layout, `kvsize` has to follow, and every rule names the code
it mirrors so the divergence is findable.

Per-cache-kind, not one formula — that distinction is the whole point:

| cache kind | grows with `num_ctx`? |
|---|---|
| `cache.NewKVCache` | yes, in 256-token blocks |
| `cache.NewRotatingKVCache(w)` | no, bounded by the window |
| `cache.NewRecurrentCache(...)` | no, constant |

A naive `layers × kv_heads × head_dim × num_ctx` over-prices gemma4 by ~5x (25 of
its 30 layers are windowed at 1024) and glm4_moe_lite by ~10x (MLA stores one
compressed latent, not per-head K and V). Over-pricing is how a model that would
have served gets refused.

**2. An explicit rung that does not fit is refused; an automatic one is
clamped.** This asymmetry is deliberate:

- An **explicit** `num_ctx` is a request. Silently serving a smaller window
  would be a lie, and the caller can act on a refusal. It names the weights, the
  KV, the rung, the headroom and what is available, and ends with the same
  phrase `runner.go:548` uses for a mid-eval OOM: *lower num_ctx or free VRAM on
  the device*. Under `requireFull` it stays `ErrLoadRequiredFull`, because
  evicting another runner raises `FreeMemory` and a retry can then succeed.
- An **automatic** `num_ctx` was never chosen by anyone. It is Ollama's
  VRAM-tier default, which is **262144** on the CUDA host — pricing that rung and
  refusing it would refuse nearly everything, and there is no knob the user
  turned that they could turn back. So the rung halves until it fits (floor 2048),
  `softContextLength` follows so the served window matches what was priced, and
  the clamp is logged. It never errors unless the weights alone do not fit.
  This is the same move `reduceAutoNumCtxForLoadOOM` makes on the llama.cpp
  path, taken *before* the load rather than after an OOM.

**3. An architecture with no rule keeps today's behaviour**, plus one warning
naming the architecture. Refusing an unpriced model would be strictly worse than
the over-admission being fixed.

**4. The operator's cap keeps its own separate check and its own plain error**
(never `ErrLoadRequiredFull` — eviction cannot lower a constant). What changed is
that the KV is now inside what `OLLAMA_MLX_MEMORY_LIMIT` bounds.

## The headroom is a placeholder, and says so

`headroom = max(512 MiB, 5% of (weights + KV))` stands in for prefill
activations, the vision tower on a multi-image request, and the transient
double-hold when a cache grows by `Concatenate`. It is **not** a model of any of
them. It is biased low on purpose: over-refusal on a serving host is worse than
over-admission, and `OLLAMA_MLX_MEMORY_LIMIT` remains the hard ceiling.

It has to be calibrated on GPU, and the estimate makes that measurable for the
first time: admission logs `MLX admission priced the context rung` with weights,
KV, headroom, rung and budget, to be compared against the runner's own
`peak memory` line.

## What this does not fix

**The KV is not the whole unpriced remainder — it is the smaller half.** For
`gemma4:26b` the estimate is 360 MiB at 8192 and 1.45 GiB at 65536, against
weights of 17–24 GB and observed peaks of 24–36 GiB. Pricing the rung makes the
ladder visible to admission and converts a mid-prefill abort into a refusal; it
does not by itself explain the peaks. The activation headroom does, and that
number is still a guess.

Fix 2 of the diagnosis — **pre-sizing the KV from `num_ctx`** instead of growing
it by `Concatenate` — is not done here. Until it is, the estimate is what the
cache *will* hold, not what it holds now, and the growth moment still transiently
holds two buffers.

## Consequences

- `NewClient(modelName, numCtx, numCtxAuto)`: the client needs to know whether
  the rung was chosen or derived. `server/sched.go` passes `req.numCtxAuto`,
  which already existed for the llama.cpp path.
- `softContextLength` became `atomic.Int64`: admission may now write it while
  `Ping` reads it.
- A load that used to be admitted and then aborted is now refused at admission,
  which is a **user-visible behaviour change** on a card that is genuinely too
  small for the requested rung. That is the intent.
- Unit-tested without a GPU (`x/mlxrunner/kvsize`, `x/mlxrunner`), including a
  ladder test that fails on the pre-change code. **The estimate has not been
  compared against a real `peak memory` yet**, and no vision-suite arm has been
  re-run to confirm no new over-refusal (acceptance criterion 5).
