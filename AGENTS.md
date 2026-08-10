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

**MLX is thread-affine.** MLX streams and their Metal command encoders are
thread-local, and an array can only be evaluated on the OS thread that built it.
Any goroutine driving MLX must call `mlx.ClaimOSThread()` once during setup, and
MLX arrays must never cross goroutines — `t.Run` subtests are separate goroutines,
so build their arrays inside the subtest. See
`docs/development.md` ("MLX threading") and `docs/maxusai/adr/0017-mlx-work-runs-on-a-permanently-claimed-os-thread.md`.
