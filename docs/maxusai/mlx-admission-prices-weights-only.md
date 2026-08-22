# The MLX runner admits a request on weight size alone, then aborts mid-prefill

MaxusAI-fork reference. Characterised 2026-08-22 on CUDA (RTX PRO 6000 Blackwell,
97.9 GB, shared with other tenants), build `0.32.14-dynres-108-g76918a7`,
MLX `0.32.1-21-g27fec90`.

> **The one thing to take away:** the parent prices a model load as
> `TotalTensorSize()` — **weights only** — so every `num_ctx` rung admits
> identically. The KV cache and the prefill activations are never in the
> arithmetic, so a rung that cannot fit is not refused at load; it is accepted
> and then killed by `cudaMallocAsync` partway through prefill. On llama.cpp the
> same request is priced before it starts.

## Symptom

`gemma4:31b-coding-mtp-bf16` (63.5 GB on disk, 1,236 tensors) running the vision
suite at `num_ctx` 8192:

```
Configured MLX memory limit from free device memory active="59.13 GiB" limit="87.51 GiB" previous="90.22 GiB"
prefix_cache.go:147 cache miss total=3846 matched=0 cached=0 left=3846
pipeline.go:125     peak memory size="70.38 GiB"
runner.go:518 ERROR Recovered a panic while evaluating a request; stopping the runner
  error="mlx: cudaMallocAsync(&data, size, stream) failed: out of memory"
sched.go:1702 WARN  runtime OOM detected; expiring loaded models to clear memory before next request
```

Then it reloads and does it again. One campaign attempt produced **13
out-of-memory aborts, 3 `an illegal memory access was encountered` aborts, 13
runner restarts and 128 prefill attempts, and not one completed cell.** Peaks
observed across those attempts: 67.09, 68.64, 69.56, 70.38, **76.76 GiB**.

The same model on a **text-only** prompt at the same `num_ctx` 8192 runs fine —
59.4 tok/s, three clean reps, draft acceptance 0.77. It is the multi-image
prefill (3,846 tokens of image embeddings) that moves the peak from ~67 to ~77
GiB. So this is not "the model is too big"; it is "the model plus *this
workload* is too big, and nothing checked".

## Cause 1 — the admission check prices weights only

`x/mlxrunner/client.go:65`:

```go
c.memory.Store(uint64(modelManifest.TotalTensorSize()))
```

That value is `modelSize`, and it is the only thing weighed at
`client.go:330` (physical shortfall) and `client.go:359` (budget shortfall):

```go
if modelSize > available { ... }
if modelSize > vramBudget { ... }
```

For the case above that is **59.13 GiB admitted against 87.51 GiB available** —
a comfortable pass — followed by a 76.76 GiB peak. The check cannot refuse a
request it should have refused, because the quantity that overflows is not in
the comparison.

Consequence for the context ladder: `8192 → 16384 → 32768 → 65536` admit
**identically** on MLX. The rung is invisible to admission. On llama.cpp the KV
for `n_ctx` is priced by the GGML estimator before the runner starts, which is
why a rung that will not fit is refused there and merely fatal here.

## Cause 2 — the KV cache is grown, not pre-sized

`x/mlxrunner/cache/kvcache.go:38` starts at `step: 256`, and `Update` grows on
demand:

```go
// Grow buffer if needed
if c.keys == nil || (prev+L) > c.keys.Dim(2) {
    steps := (c.step + L - 1) / c.step
    newKeys := mlx.Zeros(keys.DType(), B, H, steps*c.step, Dk)
    ...
    c.keys.Set(c.keys.Concatenate(2, newKeys))
```

`num_ctx` never sizes this allocation; it only bounds how far the cache is
eventually allowed to grow. Two consequences:

- **The footprint is not flat.** It climbs with the conversation, so a request
  that fits at token 500 can fail at token 3,000 in the same session.
- **Growth is a `Concatenate`**, so the moment of growth transiently holds the
  old buffer *and* the new one. The spike is largest exactly when the cache is
  already largest.

llama.cpp allocates the whole KV for `n_ctx` at load. Flat, predictable,
priceable in advance — the property this path lacks.

## Why the auto-derived limit does not save it

`configureMemoryLimit` (`runner.go:170`) samples **free device memory at load
time** and caps MLX there — the `limit="87.51 GiB"` above, against MLX's own
default of 90.22 GiB derived from *total* memory.

That is strictly better than MLX's default on a shared card, and still a race:
it is one sample, taken once, per runner. **Every unload/reload starts a new
runner subprocess with a new pid and port and takes a fresh optimistic sample**
— visible in the logs as `stopping mlx runner subprocess pid=640` /
`starting mlx runner subprocess port=40789`, each followed by its own
`Configured MLX memory limit` line. When the other tenants on the card grow after
the sample, the ceiling is already wrong and nothing re-checks it.

## The operator knob, and its measured cost

`OLLAMA_MLX_MEMORY_LIMIT` set on the **server** is not overwritten by the
per-runner derivation — `client.go:319` folds it in as
`min(derived, requested)` via `budgetWithOverride`, logs the resolution rather
than applying it silently, and refuses at `client.go:360` with
`model requires X but OLLAMA_MLX_MEMORY_LIMIT caps the MLX budget at Y` if the
value is below the weights. So it is a **stable ceiling that survives reloads**,
which is what the sampled value is not.

It is not free. `runner.go:251` records a full 12-test suite, n=2, sweeping this
exact knob:

| cell | 24 GiB | 82 GiB | delta |
|---|---|---|---|
| `scene_single` | 30.46 t/s | 30.87 t/s | +1% |
| `multi_3img` | 28.46 t/s | 29.19 t/s | +3% |
| `bbox_contract_multi` | 17.71 t/s | 27.21 t/s | **+54%** |

Peak footprint 33,536 MiB against 67,518 MiB. So a tighter ceiling **does** halve
the footprint — the allocator reuses instead of growing — and the penalty
concentrates on multi-image cells. **A single-image prompt cannot show this**;
the same comment records three sweeps that came back flat on `scene_single` and
were reported as "no throughput effect", which was true of that cell and false
of the workload the suite exists to measure.

Practical consequence: a constrained arm's **quality** metrics (IoU, recall,
contract adherence) stay comparable, and its **throughput** metrics do not.

It is a raw byte count. `"70GiB"` parses as nothing and is warned about, not
honoured; 70 GiB is `75161927680`.

## What would fix it

Two independent changes, smaller one first.

1. **Price the rung in admission.** Make the compared quantity
   `TotalTensorSize() + kvBytes(num_ctx) + activation headroom` rather than
   `TotalTensorSize()` alone. This converts a mid-prefill runner abort into the
   actionable refusal `client.go:360` already knows how to emit. It does not
   make anything fit that does not fit — it makes the failure legible and cheap
   instead of costing 13 reloads of a 63.5 GB model.

2. **Pre-size the KV from `num_ctx`.** Allocate `ceil(num_ctx/step)*step` once at
   session start instead of concatenating upward. Removes the transient
   double-buffer at growth, makes the footprint flat, and is the precondition
   for (1)'s estimate being accurate rather than approximate.

Neither is a knob; both are work in `x/mlxrunner`. Until one lands, the operable
mitigation for a large model on a shared card is `OLLAMA_MLX_MEMORY_LIMIT`, with
its throughput cost acknowledged and its arm labelled.

## Provenance and limits

- One model, one host, one campaign. The peaks are observations from a failing
  run, not a controlled sweep; no attempt was made to bisect the exact geometry
  at which 8192 stops fitting.
- The 3 `illegal memory access` aborts are a different failure class from the 13
  OOMs and are **not** explained here. They appeared only under memory pressure.
  Whether they are a consequence of the OOM path or an independent MLX/CUDA
  defect is unestablished.
- The 24/82 GiB throughput table is quoted from `runner.go:251`, measured on
  `gemma4:31b-nvfp4`, not re-measured here.

Related: [`format:"json"` disables speculative decoding on the MLX runner](mlx-constrained-decode-disables-speculation.md)
and [grammar-aware speculation is correct and inert](grammar-speculation-measured-inert.md)
are the other two standing MLX-runner findings. The sibling trap to this one is
the VRAM-derived **context** default (`default_num_ctx` 262144 on this host,
which collapses decode ~25x): both come from deriving a number once, from device
memory, and treating it as safe for the workload that follows.
