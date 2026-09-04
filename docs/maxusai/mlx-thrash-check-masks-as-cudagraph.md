# MLX's graph-cache thrashing check kills the runner, and the log blames `cudaGraphAddDependencies`

MaxusAI-fork reference. Characterised 2026-08-22/23 on CUDA (RTX PRO 6000
Blackwell), build `0.32.14-dynres-108-g76918a7`, MLX `0.32.1-21-g27fec90`, and
reproduced on the pre-merge image `0.32.14-rc0-dynres-20-geb0ad43` (MLX `adf21dea`).
Upstream: ml-explore/mlx [#4326](https://github.com/ml-explore/mlx/issues/4326)
(open) and its fix [#4356](https://github.com/ml-explore/mlx/pull/4356) (open,
mergeable, unmerged as of 2026-08-23).

> **The one thing to take away:** what the log shows —
> `mlx: cudaGraphAddDependencies(...) failed: invalid argument` — is the **second**
> failure, not the first. The first is MLX's graph-cache *thrashing check*, a
> performance advisory implemented as a `std::runtime_error` that fires once a
> runner has ever seen more than 2 × `MLX_CUDA_GRAPH_CACHE_SIZE` distinct graph
> shapes. It kills the request; ollama's deferred cleanup then trips over the state
> it left behind and panics again, and Go's `recover()` keeps only that later
> panic. The message you would grep for never reaches the log. This is a
> pre-existing upstream bug, not a regression from the v0.32.15 sync, and it is
> why every MLX think=on cell in the 2026-08-22 campaign carries error blocks.

## Symptom

The vision-suite campaign on `vsuite` (`sync-0.32.15`) over ~7 hours of MLX
nvfp4 serving: **99 occurrences**, all on MLX models, none on llama-server.
They landed in the banked scores as error blocks:

| cell | error blocks |
|---|---|
| `gemma4:12b-nvfp4` think=on | **17 of 27** |
| `gemma4:26b-nvfp4` think=on | 2 of 27 |
| `gemma4:31b-nvfp4` think=on | 1 of 27 |
| `qwen3.8:27b-nvfp4` think=on | 3 (cell interrupted) |
| every think=false cell, every GGUF cell | 0 |

They come in clusters — one abort, a runner restart, a run of clean requests,
another abort — and the standalone finetext probe eventually took one as a 500
and `run_engine_compare.sh`'s `set -eu` ended the campaign. And the string
`Cache thrashing` appears **zero** times in the server log.

## Mechanism, in three parts

### 1. The check is cumulative and it throws

`mlx/backend/cuda/lru_cache.h` @ `27fec909`:

```cpp
LRUCache(const char* env_name, int default_capacity)
    : LRUCache(env::get_var(env_name, default_capacity)) {
  if (env::get_var("MLX_ENABLE_CACHE_THRASHING_CHECK", 1)) {
    env_name_ = env_name;
  }
}
...
if (env_name_ && ++cache_misses_ > 2 * capacity_) {
  throw std::runtime_error(fmt::format(
      "Cache thrashing is happening, please set the environment variable "
      "{} to a larger value than {} to fix degraded performance.", env_name_, capacity_));
}
```

`cache_misses_` is a **lifetime counter that is never reset**. Once a process has
ever accumulated more than `2 × capacity` (default `MLX_CUDA_GRAPH_CACHE_SIZE=400`,
so 800) distinct graph-key misses, *every subsequent miss throws*. It is not a
rate check; it is a fuse.

### 2. The throw leaves the encoder poisoned (upstream #4326)

`mlx/backend/cuda/device.cpp::CommandEncoder::commit()` @ `27fec909`, in order:

| offset | what |
|---|---|
| +9 / +16 | `cudaGraphAddDependencies(graph_, from_nodes_, to_nodes_, …)` — dependencies recorded |
| +29 | `auto& graph_exec = graph_cache_[graph_key];` — **the throw above fires here** |
| +61…+67 | `from_nodes_.clear(); to_nodes_.clear(); node_map_.clear(); graph_ = CudaGraph(device_); …` — **reset, on the success path only** |

So a thrash throw at +29 returns with `from_nodes_` / `to_nodes_` / `node_map_` /
`graph_` still populated with the failed graph. The next `commit()` mixes those
stale handles into a fresh graph and `cudaGraphAddDependencies` rejects them with
`cudaErrorInvalidValue` — *invalid argument*. That is #4326 verbatim; #4356 fixes
it by splitting `commit_impl()` and resetting on any exception.

### 3. Ollama masks the first panic with the second

`mlx.Eval` → `mlxCheck` panics with the thrash message. The request pipeline has
`defer session.close()` (`x/mlxrunner/pipeline.go:131`), and `close()` calls
`mlx.AsyncEval(arrays...)` (`prefix_cache.go:588`) — a second commit, on the
poisoned encoder — which panics with `cudaGraphAddDependencies`. The MLX worker's
recover site (`x/internal/mlxthread/thread.go`):

```go
func run(fn func() error) (res result) {
	defer func() {
		if v := recover(); v != nil {
			res.panic = &panicError{value: v, stack: debug.Stack()}
		}
	}()
```

`recover()` returns the **most recent** panic. The thrash message is gone; the
stack shows the tell-tale double `panic(` — `prefill → doEval → panic`, then
`cacheSession.close → AsyncEval → doEval → panic` — and the logged error is the
second one. Every one of the 99 campaign stacks has that shape.

The runner then stops (`runner.go:518`), the parent restarts it, the counter is
zero again, and the next ~800 misses run clean. Hence clusters rather than a
permanent wedge — the restart is the only thing bounding it.

## Why think=on, and why MLX only

The counter counts *distinct graph shapes*. A think=on cell generates thousands of
decode steps across growing KV lengths and climbs the `num_ctx` ladder; a
think=false cell at `num_ctx` 8192 never accumulates 800 distinct keys before
`RESTART_CMD` gives it a fresh runner. llama-server does not use MLX at all, so
GGUF cells cannot see it — which is also why the three GGUF teachers and all five
GGUF roster models came through with zero error blocks.

## Reproduction — from a cold start, every time

Driver: [`vision-suite/mlx_thrash_probe.py`](vision-suite/mlx_thrash_probe.py) — N text-only `/api/generate` requests against
`gemma4:12b-nvfp4`, each with a **distinct prompt length** (`"Count: 0 1 2 … i"`),
`num_predict=1`, `num_ctx=8192`, cold container each phase.

| phase | image | settings | result |
|---|---|---|---|
| A | new (`27fec909`) | `MLX_CUDA_GRAPH_CACHE_SIZE=8` | **120/120 fail from request #1**. Server: `thrashing=372`, `cudaGraph=4`, 120 runner restarts |
| B | new | **defaults** (cache 400, check on), 1000 requests | clean for 707 with latency climbing 0.2 → 11.4 s (each new length is a miss), **request 708 → `500-cudaGraph`**, then 292 clean on the fresh counter. Server: `cudaGraph=3`, **`thrashing=0`** |
| C | old (`adf21dea`) | `MLX_CUDA_GRAPH_CACHE_SIZE=8` | **120/120 fail from request #1**. Server: `thrashing=372`, `cudaGraph=3`, 120 runner restarts — **the pre-merge build has the full signature, masked second failure included** |

Phase B is the campaign's signature reproduced on demand: the visible error is
`cudaGraph`, the thrash text is absent, and the server stack is the same double
panic. Phase A's `cudaGraph=4` against 120 throws shows the masked second failure
is stochastic — it needs the deferred `close()` to actually commit — which is why
the campaign saw *mostly* clean restarts punctuated by error blocks rather than a
failure on every throw.

Two earlier probes (6 identical requests; 12 varied prompts/images) were clean on
both images. They never generated enough distinct shapes. **A reproduction needs
hundreds of distinct graph keys in one process**, not a particular request.

## Not a regression from the v0.32.15 sync

The check was added by ml-explore/mlx #2600, merged **2025-09-19**. Both of our
pins — `adf21dea` (pre-merge) and `27fec909` (post-merge) — carry it in
`lru_cache.h`, and Phase C reproduces the throw on the pre-merge image. The MLX
bump in #208 did not introduce this.

## Mitigation

MLX reads three knobs in our pinned `device.cpp`, and `x/mlxrunner/client.go:388`
passes `os.Environ()` to the runner subprocess, so a container env reaches MLX:

| knob | effect | cost |
|---|---|---|
| `MLX_ENABLE_CACHE_THRASHING_CHECK=0` | the fuse never blows; the LRU still evicts | none — it only disables an advisory |
| `MLX_CUDA_GRAPH_CACHE_SIZE=<larger>` | more misses before the fuse | graph memory; only delays it |
| `MLX_USE_CUDA_GRAPHS=0` | no graph commit at all | decode throughput |

Measured under the same cold-start recipe:

| phase | image | settings | result |
|---|---|---|---|
| D | new | `MLX_CUDA_GRAPH_CACHE_SIZE=8` + `MLX_ENABLE_CACHE_THRASHING_CHECK=0` — Phase A's conditions with the check off | **120/120 ok, 0 fail**, 0.2–0.3 s/request. Server: `cudaGraph=0`, `thrashing=0`, 1 runner start |
| E | new | **defaults** + `MLX_ENABLE_CACHE_THRASHING_CHECK=0`, 400 unique-length requests | **400/400 ok, 0 fail**. Server: `cudaGraph=0`, `thrashing=0`, 1 runner start |

Phase A vs Phase D is the whole argument in one row pair: identical container,
identical model, identical 120 requests, cache forced to 8 — **120/120 fail with
the check on, 120/120 pass with it off**, and no slower. It also settles the one
thing the logs could not show: the masked first panic was the thrash throw,
because removing only that throw removes the `cudaGraph` failure entirely.

Worth stating plainly: even once #4356 lands upstream, ollama still loses the
request, because the throw reaches Go and the runner restarts. A *performance
advisory* surfaced as an *exception* is the wrong trade for a server that cannot
catch-and-continue. Disabling the check is not hiding a fault; it is declining to
turn a slow path into a dead request.

## What would fix it in the fork

> **Status 2026-09-04:** both proposals shipped in #212 (merged 2026-08-26) and have
> run every campaign since (`sync15nt`, `mlx0330nv`, `mlx0332nv`: ~14 h and two full
> fleets, zero thrash aborts). This document is the source of record the code cites.
>
> **Implemented in #212** (branch `fix/mlx-thrash-check-default-off`): the runner
> subprocess now starts with `MLX_ENABLE_CACHE_THRASHING_CHECK=0` unless the operator
> exported the variable (`x/mlxrunner/client.go`, `mlxRunnerEnvDefaults`), and the
> pipeline's deferred cleanups go through `guardClose` (`x/mlxrunner/unwind.go`) so a
> cleanup that fails during unwinding no longer replaces the first panic in the log.
> Images built before that commit (including `sync-0.32.15`) still need the env set
> on the container.

1. **Set `MLX_ENABLE_CACHE_THRASHING_CHECK=0` for the MLX runner subprocess by
   default** (next to the `CUDA_PATH` / `CUDA_HOME` `setEnv` calls in
   `x/mlxrunner/client.go`), overridable by an operator who wants the advisory.
   One line; removes the failure entirely.
2. **Log the first panic, not just the last.** `mlxthread.run()` could capture the
   original panic value when a deferred cleanup panics during unwinding, so the
   log names the real cause. This does not change behaviour; it makes the next
   person's grep work.
3. Separately, the `RESTART_CMD`-per-cell discipline already bounds the blast
   radius to one request per fuse; a per-*request* runner restart would not be
   acceptable, so (1) is the actual fix.

## Provenance and limits

- The text of the *first* (masked) panic is inferred from the mechanism and the
  cumulative-counter arithmetic (~1.1 misses/request × 708 ≈ 800), not read from
  a log — by construction it cannot be. Phases D/E test that inference directly:
  with the check disabled the failure should vanish under identical conditions.
- One model (`gemma4:12b-nvfp4`), text-only driver; the campaign failures were on
  image-bearing requests. The mechanism is shape-count-driven, not
  modality-driven, and the stacks are identical, but the driver is a proxy.
- Phase A/C's per-request runner restart (11–12 s) is ollama's abort-and-reload,
  not the model; do not read those latencies as anything else.

Related: [the MLX runner admits on weight size alone](mlx-admission-prices-weights-only.md)
is the other failure seen in the same campaign — an OOM class, unrelated to
this one and not to be conflated with it.
