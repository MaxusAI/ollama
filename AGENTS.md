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

**Serving a benchmark sweep requires `OLLAMA_MAX_LOADED_MODELS=1`.** The
scheduler keeps every model it has served resident, so a sweep over N models
holds N models at once — count, not size, is what exhausts the host. Measured
2026-08-17 on a 128 GB Mac: a four-arch preflight ladder left three runners
resident at 68.7 + 22.0 + 20.7 GB, reaching 106 GB used and 53.9 GB of swap
while Docker and two other VMs were live, one allocation from OOM-killing
unrelated work. Check free memory before loading a >30 GB build
(`gemma4:31b-mlx-bf16` is 63.5 GB, `nemotron3:33b-bf16` is 66.1 GB), and treat
"skip model X, not enough headroom" as standing until withdrawn.
See `docs/maxusai/spec/apple-silicon-build.md`.

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

**Always let the context ladder run, and read the rung it stops at as a result.**
`num_predict` is derived as `num_ctx - CTX_PROMPT_RESERVE`, so fixing the window
silently fixes the generation budget too. The 16384 start rung yields exactly
8192 generated tokens, and that is not a safe default: measured 2026-08-18,
`nemotron3:33b-q4_K_M` think-on needs 8385 / 10226 / 4127 tokens to terminate, so
at the start rung it caps in about half its runs and returns `done_reason=length`
with **zero characters of answer** and 24–27k characters of unclosed thinking.
Given a rung that derives a real budget it answers every question correctly, 3/3.
The context was never short — `prompt_eval_count` 6203, a 131072 window at 12%
occupancy. Two things follow, and neither is optional:

- **The converged rung IS the measurement** of the window that model needs, and
  it belongs in the write-up. A run that never escalated did not measure it.
- **req/h and tok/s come only from a cell that terminated** (`done_reason=stop`).
  A capped cell's `eval_count` is the cap, so a rate derived from it measures the
  harness setting and moves 2.3x when the ladder climbs a rung. `was_capped()`
  marks these and summarizers render `capped` instead of a number; never quote one.

See `docs/maxusai/spec/vision-harness-reuse.md` (H1–H11, H4a for the ladder, H9 single request path, H10 cold start/residency, H11 host+build provenance),
`docs/maxusai/adr/0028-one-runner-one-set-of-helpers.md`,
`docs/maxusai/adr/0022-thinking-is-off-for-vision-work.md` (the three traps), and
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
