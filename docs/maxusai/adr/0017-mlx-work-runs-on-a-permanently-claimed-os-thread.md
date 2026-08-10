# ADR 0017: MLX work runs on a permanently claimed OS thread, and arrays never cross goroutines

- **Status:** accepted (2026-08-10)
- **Date:** 2026-08-10
- **Deciders:** MaxusAI fork maintainers

## Context

`x/mlxrunner` panicked intermittently with

```
mlx: There is no Stream(gpu, 0) in current thread. at mlx/c/transforms.cpp:73
```

The failing test name varied run to run and always landed in a pre-existing
speculative-decode test, which made it look like inherited noise. It is not.
Measured 2026-08-10: `main` (08d3784c) was clean (0/8), while
`fix/review-findings-2026-08-10` — whose changes are unrelated to threading —
failed 4/20. `GOGC=off` made it vanish. The rate tracked scheduler perturbation,
not any one change.

This is an inherited upstream behaviour change, not a bug that was always latent.
MLX made each thread own its default stream in **v0.31.2** (`df7f7db94`, upstream
PR ml-explore/mlx#3281, named in the v0.31.2 release notes); before that the
default stream lived on the process-global `Scheduler` (`mlx/scheduler.h`
`default_streams_`). This repo crossed that boundary in `534342e7` (2026-05-03,
upstream PR ollama/ollama#15845), which bumped `MLX_VERSION` from `38ad257088`
(v0.31.1-23, pre-thread-local) to `e8ebdebeeb` (v0.31.2-7, post). We now pin
`973e27f82` (v0.32.0-21, `version.h` 0.32.1-dev) with `mlx-c` at `fba4470b`
(v0.6.0-4).

Before that bump `DefaultStream()` was a `sync.OnceValue` — resolve once per
process — which was *correct* against pre-0.31.2 MLX. The uncomfortable detail:
the same commit that crossed the boundary also shipped upstream's partial fix,
and it converted that `OnceValue` into a resettable process-global cache. It kept
a process-global holding a value that had just become thread-local, so the
residual defect was carried forward by the upstream fix rather than introduced
here.

Two facts about MLX force the decision:

- The default stream for each device lives in thread-local storage
  (`mlx/stream.cpp`, `default_stream_storage` is `static thread_local`), and the
  Metal command encoder behind it lives in a `thread_local` map
  (`mlx/backend/metal/device.cpp`, `get_command_encoders`). `get_command_encoder`
  throws the message above when the encoder is absent from the *calling* thread's
  map. The default **device**, by contrast, is a process-wide singleton
  (`mlx/device.cpp`, `mutable_default_device`).
- Every array records the stream it was built on, so an array can only be
  evaluated on the thread that built it. `mlx-c` wraps no thread-portable stream
  (`new_thread_unsafe_stream`, whose encoder would land in MLX's process-global
  fallback map, is not exposed), so there is no way to opt out of this.

Two defects followed from that, and both had to go:

1. `DefaultStream()` (`x/mlxrunner/mlx/stream.go`) cached MLX's thread-local
   stream in a *process-global*. That is what put `Stream(gpu, 0)` — the first
   thread's handle — into every panic: later threads were handed a stream they
   could never evaluate on.
2. Nothing pinned the goroutines that were *not* the runner worker. `mlxCall`
   (`x/mlxrunner/mlx/mlx.go`) locked and unlocked the OS thread around a *single*
   C call, which cannot hold an invariant that spans calls. To be fair to it, that
   lock was never meant to: it was added in `d3e67e30` (2026-04-13) for the
   `__thread` error-message buffer, three weeks before the repo crossed the
   thread-local boundary. It was correct for its stated purpose and merely
   insufficient for one it was never written to serve.

`mlxCall` was not a usable choke point either way: it wraps 9 of the ~264
`C.mlx_*` invocations in the package, and the graph-construction ops in `ops.go`,
`ops_extra.go` and `gated_delta.go` bind their stream via `DefaultStream()`
without ever passing through it.

**What was already protected.** The runner worker was *not* unpinned:
`x/internal/mlxthread` locks its worker goroutine and deliberately never unlocks
(`thread.go`), and `x/create`'s worker did the same with a bare
`runtime.LockOSThread()`. Both arrived with upstream's `534342e7`. The observed
crashes were in **tests**, whose goroutines were unpinned and inherited the cached
stream. So worker pinning was necessary and already present; what it could not fix
was the process-global cache.

That also explains why the Python guidance for this same upstream change
("initialize the stream inside the worker thread") does not transfer. A
`threading.Thread` *is* an OS thread, so initializing per worker is a complete fix
in Python. In Go it is only half: the pin must be permanent, and the cache must
not outlive the thread that filled it.

## Decision

Add `mlx.ClaimOSThread()` and require every goroutine that drives MLX to call it
once during setup, before its first MLX operation.

1. **The claim is permanent and idempotent.** It calls `runtime.LockOSThread()`
   and never unlocks — MLX thread-local state outlives any single call, so the
   thread belongs to that goroutine until it exits. Idempotency uses a C
   `__thread` flag rather than goroutine-local storage (Go has none): the Go
   runtime only ever schedules a locked goroutine on its own thread, so "this
   thread is claimed" implies "this goroutine claimed it". The flag is set *after*
   the lock, since the goroutine may still migrate between the check and the lock.
2. **Claiming resets the cached default stream**, so a new owner resolves its own
   instead of inheriting a dead or foreign one.
3. **The hot path is untouched.** `DefaultStream()` stays a plain cached field
   read; the claim happens at setup and in the low-frequency `mlxCall` path.
4. **Arrays never cross goroutines.** This is the invariant the type system cannot
   express: `t.Run` subtests are separate goroutines, so a test that builds MLX
   fixtures in the parent and evaluates them in subtests is invalid. Fixtures are
   built inside the subtest.

Callers: `x/mlxrunner/server.go` and `x/create/mlxthread.go` worker init, plus the
`skipIfNoMLX` helper in every package whose tests drive MLX.

## Alternatives considered

- **Per-operation thread-affinity check** (a cgo `__thread` read inside
  `DefaultStream()` or `New()`, so any goroutine self-pins on first use). Correct
  for any caller with no setup discipline, and it was the tempting option. Rejected
  on measurement: on an M5 Max one real graph-construction op is **224 ns**, a cgo
  TLS read is **22.5 ns**, and the current cached `DefaultStream()` is **2.2 ns** —
  about a 10% tax on graph construction, in a fork that tracks req/h (ADR 0012).
- **Unconditional `runtime.LockOSThread()` per call, never unlocked.** No cgo flag
  needed, but the runtime's nesting counter is a `uint32` that panics on overflow;
  a long-running server doing tens of thousands of ops/second reaches 2³² in about
  a day.
- **Funnel every MLX operation to one dedicated thread through a command queue.**
  The only design that needs no caller discipline and would also permit arrays to
  cross goroutines. Rejected for now: it means routing ~250 direct cgo call sites,
  and a channel round-trip per 224 ns op unless every site takes a fast path that
  costs a cgo check anyway. Reconsider if callers keep getting the discipline wrong.
- **One process-global stream via `new_thread_unsafe_stream`.** Would remove the
  affinity requirement outright, since its encoder lands in MLX's process-global
  fallback map. Unreachable: the symbol appears nowhere in `mlx-c`, and this is
  not a stale-bindings problem — our generator wraps the whole `mlx-c` surface
  (614 declared, 614 wrapped, exact bijection), so `go generate` recovers nothing
  and a bindings bump would not either. It needs a patch to `mlx-c` itself. Even
  then it trades a loud, immediate error for silent corruption unless we add the
  mutex MLX deliberately omits.
- **Pin the vendored MLX back to before streams became thread-local.** This is a
  trap rather than a non-starter: every C symbol the repo uses exists at the
  0.31.1-era pin, so it would *compile*. It also reverts 248 MLX commits, 63 of
  them in the Metal backend — RoPE, nvfp4 split-K, quantized-matvec and gather-MM
  correctness fixes, and the NAX chain — to avoid a bug that is ours to fix, and
  only defers it to the next bump.

## Consequences

- Positive: thread affinity is structural rather than probabilistic. Verified with
  GC on: the deterministic reproducer failed 5/5 before and passes after;
  `./x/mlxrunner/` is 25/25 on this branch and 25/25 on the branch that was 4/20,
  and 12/12 under `GOGC=1`. `GOGC=off` is no longer needed.
- Positive: cross-goroutine MLX sharing now fails **immediately and always**
  instead of flaking. That surfaced six latent test bugs (parent-built fixtures
  consumed by `t.Run` subtests) in `x/mlxrunner/cache` and `x/models/nn`, all fixed
  here.
- Negative: a goroutine that forgets to claim gets MLX's cryptic message rather
  than a named one, and the discipline is convention, not a compile-time check.
- Negative: each claiming thread creates its own MLX stream holding a Metal command
  queue, and a test binary claims once per MLX test (42 sites in
  `x/mlxrunner/cache`). Measured, this is not a practical ceiling: 300 sequential
  claims produced 300 streams with no failure, and the device accepted 4096
  simultaneous command queues. Thread exit is the release — pthread TSD destructors
  run when a locked goroutine exits, so live encoders stay bounded. Wrapping
  `mlx_clear_streams` would *not* help: `gpu::clear_streams` clears only the calling
  thread's encoder map and never shrinks `all_streams()`.
- Negative: the ownership flag is only sound while the pin is never released.
  Pairing `ClaimOSThread` with `runtime.UnlockOSThread` would return a thread to
  the pool still marked as owned, and the next goroutine scheduled onto it would
  skip claiming one of its own.
- Follow-up: `x/imagegen` is a second, independent MLX binding (its own cgo
  preamble and generated wrappers, its own worker) that does not route through
  `ClaimOSThread` — so the invariant in `AGENTS.md` is not enforceable there today.
  It keeps a package-global stream (`x/imagegen/cmd/engine/generate.go`
  `generationStream`) and its tests pair `LockOSThread` with `UnlockOSThread` in
  cleanup. It survives on process separation (the two runners are never
  co-resident, and it has no process-global default-stream cache), so this is an
  unaudited surface rather than a known defect. Worth checking whether its
  long-standing SIGSEGV is the same bug class.

## Conformance

- `TestMLXOperationsSurviveRescheduling` (`x/mlxrunner/mlx/thread_test.go`) drives
  MLX from a plain goroutine while every P is saturated and the goroutine yields
  between operations. It is load-bearing: with `ClaimOSThread` stubbed out it
  reproduces the exact panic.
- `x/internal/mlxthread` `TestDoUsesSameOSThread` continues to pin the worker
  contract.
- Full `./x/...` (excluding the pre-existing `x/imagegen` SIGSEGV) green over 6
  consecutive sweeps; every test in every claiming package also passes when run
  individually, since a panic aborts the process and would otherwise mask later
  failures.
