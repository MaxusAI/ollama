# AGENTS.md

## Building

For a full build from the repository root:

```sh
cmake -B build .
cmake --build build --parallel 8
./ollama serve
```

For quick Go-only iteration against an existing native payload:

```sh
go build .
go run . serve
```

See `docs/development.md` for prerequisites, platform notes, GPU backends, and
the full development workflow.

## Invariants

**MLX is thread-affine.** MLX streams and their command encoders are thread-local,
and an array can only be evaluated on the OS thread that built it. So a goroutine
driving MLX must stay on one OS thread, and MLX arrays must never cross
goroutines — `t.Run` subtests are separate goroutines, so build their arrays
inside the subtest. Never cache a stream anywhere that outlives its thread.

There are two independent MLX bindings, and they enforce this differently:

- `x/mlxrunner/mlx` — call `mlx.ClaimOSThread()` once during setup. It pins the
  goroutine permanently and resets the Go-side stream cache for the new owner.
- `x/imagegen/mlx` — no claim call; its stream cache is thread-local in C, so each
  thread resolves its own. Callers still pin (`InitMLX` locks the main goroutine).

See `docs/development.md` ("MLX threading"),
`docs/maxusai/adr/0017-mlx-work-runs-on-a-permanently-claimed-os-thread.md`, and
`docs/maxusai/adr/0018-imagegen-caches-mlx-streams-per-thread.md`.
