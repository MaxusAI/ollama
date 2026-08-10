# ADR 0018: `x/imagegen` caches MLX streams per OS thread instead of claiming the thread

- **Status:** accepted (2026-08-11)
- **Date:** 2026-08-11
- **Deciders:** MaxusAI fork maintainers
- **Related:** [ADR 0017](0017-mlx-work-runs-on-a-permanently-claimed-os-thread.md)
  (the same invariant, solved differently in `x/mlxrunner`)

## Context

[ADR 0017](0017-mlx-work-runs-on-a-permanently-claimed-os-thread.md) established
that MLX streams and their command encoders are thread-local, so an array can
only be evaluated on the thread that built it. It fixed `x/mlxrunner/mlx` and
left `x/imagegen` — a second, independent MLX binding with its own cgo layer and
generated wrappers — recorded as an unaudited surface.

`x/imagegen/mlx` and `x/imagegen/nn` had failed with a SIGSEGV for long enough to
be treated as baseline noise and excluded from every verification sweep. They were
the same bug.

`x/imagegen/mlx/mlx.go` cached the default GPU and CPU streams in **process-global**
C statics, filled lazily by whichever thread got there first:

```c
static mlx_stream _default_stream = {0};   // and _cpu_stream
```

Both `mlx_default_gpu_stream_new()` and `mlx_default_cpu_stream_new()` resolve
`mlx::core::default_stream()`, which is thread-local — so the cache handed the
first thread's stream to every later thread. `useMLXTestThread` pairs
`runtime.LockOSThread` with `runtime.UnlockOSThread` in cleanup, so each test ran
on a different thread and inherited a stream it could not use.

Confirmed by controlled experiment (2026-08-11): every test passed in isolation
and any two in sequence crashed; two probes differing *only* by re-resolving the
stream on the current thread gave SIGSEGV versus pass, with the cleanup's
`ReleaseAll()`/`ClearCache()` running in both arms — which rules out a lifetime
bug.

It surfaced as a segfault rather than a panic because this binding installs no
error-capturing handler: the failed eval left the array unallocated, and
`mlx_array_data_float32` dereferenced null inside MLX.

## Decision

Make the two cached streams thread-local (`static __thread`), so each OS thread
resolves its own. Nothing else changes: the lazy-resolve shape, the call sites,
and `set_default_stream` all stay as they were.

`__thread` is already used in this tree's cgo (`x/mlxrunner/mlx/mlx.go`) and is
supported by every compiler we build with, including mingw on Windows.

## Alternatives considered

- **Port `ClaimOSThread` from `x/mlxrunner`** — the mechanism ADR 0017 chose.
  Rejected because it would not have fixed this crash: `x/imagegen`'s callers
  *already* pin (`InitMLX` locks the main goroutine, the test helper locks per
  test). The missing piece was never the pin, it was the cache, and a claim API
  on top of a process-global cache still hands thread one's stream onward.
- **Delete the cache and call `mlx_default_gpu_stream_new()` per operation.**
  Correct, and it removes the failure mode by construction, but it adds a cgo call
  to every op on the image path for a value that never changes within a thread.
- **Keep the global and require all MLX work on one thread.** This was the de
  facto status quo, and it is precisely what broke: the test helper releases the
  pin in cleanup, so "one thread" was never actually guaranteed.

## Consequences

- Positive: `x/imagegen/mlx` and `x/imagegen/nn` pass. They had been excluded from
  every sweep as inherited noise, so this restores real coverage rather than just
  removing a crash.
- Positive: `set_default_stream` is now per-thread, matching `mlx_set_default_stream`,
  which was already thread-local. The two previously disagreed about scope.
- Negative: the two bindings now solve one invariant two ways — `x/mlxrunner` pins
  the goroutine and resets a Go-side cache on claim, `x/imagegen` keeps the cache
  per thread in C. The asymmetry is forced: `x/imagegen`'s cache lives in C where
  `__thread` exists, while Go has no goroutine-local storage, so the Go-side cache
  had to be tied to an explicit claim instead.
- Negative: one stream handle is leaked per thread that touches MLX, where before
  it was one per process. Bounded by thread count and negligible.
- Follow-up: `go vet` still flags `x/imagegen/mlx/compile.go:84` for `unsafe.Pointer`
  misuse. Pre-existing, unrelated, untouched here.

## Conformance

- `TestDefaultStreamIsPerThread` (`x/imagegen/mlx/mlx_test.go`) runs the same
  build/eval/read sequence on two OS threads in turn, each goroutine locking a
  thread and then exiting so the runtime tears it down. It is load-bearing:
  reverting `__thread` reproduces the SIGSEGV.
- `./x/imagegen/mlx/` and `./x/imagegen/nn/` green, including the previously
  crashing `TestKeptSurvives` and `TestBasicCleanup` in the same run.
