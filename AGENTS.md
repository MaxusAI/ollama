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

**The benchmark harness has one runner and one set of helpers.** Do not write a
script that loops over models — `docs/maxusai/vision-suite/run_engine_compare.sh`
is the only entry point that climbs the `num_ctx` ladder, derives `num_predict`
for think-on, restarts the server per cell and stamps the power mode. A repeated,
subsetted or sampling-overridden **arm** is environment knobs (`REPEATS`,
`TAG_PREFIX`, `ONLY_TESTS`, `TEMPERATURE`), not a new file; a missing capability
is a patch to the runner. Summarizers import `engine_for` / `was_capped` /
`ctx_for` / `tag_for` / `load` from `summarize_engine_compare.py` rather than
redefining them, and tables are pasted verbatim from a generator — never
retyped or reformatted.

Six bespoke loops were written in one week and not one climbed the ladder, so
every one measured think-on at the think-off cap and published empty responses
as results. A re-implemented `was_capped` used `==` where the original uses
`>=`. A hand-copied table dropped `num_ctx` and published a scene IoU of 0.000
that was really 0.872.

See `docs/maxusai/spec/vision-harness-reuse.md` (H1–H8),
`docs/maxusai/adr/0028-one-runner-one-set-of-helpers.md`, and
`docs/maxusai/adr/0012-benchmark-report-templates.md`.

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
