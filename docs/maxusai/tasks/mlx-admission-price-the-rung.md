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
   llama ones. ☑ **Compared to a measured load 2026-09-05**: five models × four rungs × two
   request shapes against the runner's own `peak memory` line, 40/40 clean — see "What the
   GPU phase measured" below. The KV term is small and the peaks are flat across rungs; the
   remainder is the prefill transient, now calibrated per architecture.
2. ☑ Admission compares weights + KV(rung) + headroom, and the four ladder rungs
   admit differently — `TestLadderRungsAdmitDifferently` fails on the pre-change
   code (`num_ctx 16384 needs 42949672960, not more than 42949672960 at 8192`).
3. ☑ A refusal in `client.go`'s existing shape, naming weights / KV / rung / headroom /
   available and ending with the `runner.go:548` phrase `lower num_ctx or free VRAM on
   the device`.
4. ☐ KV preallocated from `num_ctx` at session start (#210 fix 2). **Not started.**
   Until it lands the estimate describes what the cache *will* hold, and the
   `Concatenate` growth moment still transiently holds two buffers.
5. ☑ (by arithmetic and by the calibration run) / ☐ (campaign) No new over-refusal: with the
   calibrated headroom every vision-suite model admits at every ladder rung on the CUDA host —
   worst case gemma4:31b at 65536 needs 17.3 + 5.8 + 14.5 = 37.6 GiB and qwen3.6 about 39 GiB
   against the 45.7 GiB budget the campaigns run with (16 GiB overhead) and 61.7 GiB uncapped.
   The calibration itself loaded all five models at all four rungs under the new admission.
   A think-off campaign on the calibrated binary has not been run.
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

### The headroom, calibrated (2026-09-05)

The placeholder `max(512 MiB, 5% of (weights + KV))` was 10–25× too small. The GPU phase
showed the remainder is the **prefill transient** — activations for one 2048-token prefill
chunk plus the vision tower — which does not depend on `num_ctx` and saturates once a
prompt fills a chunk. So the headroom is a per-architecture constant (measured peak − weights
at a full chunk, plus ~10%): **gemma4 14.5 GiB, qwen3.5 dense 10.5 GiB, qwen3.5-MoE 16 GiB,
unknown architectures max(10 GiB, 5%)**. `admissionHeadroom(arch, weightsPlusKV)` in
`client.go` carries the table and the measurements. Lowering `num_batch` shrinks the
transient (it bounds the chunk, the GGML `num_batch` analogue) and is not modelled.

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

### What the GPU phase measured (2026-09-05, CUDA host, `sync-0.33.3` + this branch's binary)

`gate-gpu-276b.sh`: five models × rungs 8192 / 16384 / 32768 / 65536 × two request shapes
(one 2048×1152 ladder image; three ladder images), `OLLAMA_GPU_OVERHEAD` 16 GiB (budget
45.7 GiB), a fresh runner per rung so every cell has its own admission line. 40/40 clean.
Data: `preflight/runs/gpu276-calibration-2026-09-05.jsonl`.

| model | weights | KV @8192 | KV @65536 | peak, 1 image | peak, 3 images | peak − weights, 1 img | 3 img |
|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | 7.1 GiB | 0.44 | 1.30 | 9.53 | 16.80 | 2.4 | **9.7** |
| gemma4:26b-nvfp4 | 16.3 | 0.35 | 1.40 | 25.65 | 26.54 | 9.4 | **10.2** |
| gemma4:31b-nvfp4 | 17.3 | 1.40 | 5.80 | 25.37 | 30.36 | 8.1 | **13.1** |
| qwen3.8:27b-nvfp4 | 16.9 | 0.64 | 4.10 | 26.12 | 26.12 | 9.2 | **9.2** |
| qwen3.6:35b-a3b-nvfp4 | 22.0 | 0.22 | 1.30 | 29.26 | 36.27 | 7.3 | **14.3** |

Findings against the list below: (1) **the peak was identical at all four rungs for every
model** — the KV is consumed by tokens actually processed, so the rung is a cap admission
assumes, not where the memory goes; the remainder is set by the largest prefill chunk (gemma4
sees 1122 tokens for one image, a partial chunk, and 3337 for three; qwen sees 2325 and 3060,
so one and three images cost the same there) and grows with the model; on the MoE it also
moves with generation length (28.6–36.3 GiB peaks for the same prompt). (2) Layer counts
were logged on every admission line; not cross-checked against `NewCaches` at runtime beyond
the unit tests. (3) No over-refusal at any rung, see criterion 5. (4) The sliding transient is
inside the measured peaks and therefore inside the constant; no separate term. (5) The auto
clamp was not exercised on a real load (every calibration request set `num_ctx`) — still ☐.
Two probe bugs cost two attempts before this data existed: the ladder images live under
`preflight/ladderimgs/`, and a ~260 KB request body cannot be passed as a curl argument
(Linux caps one argv string at 128 KiB; send it with `--data-binary @file`).

The original measurement plan, kept for the record:

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
