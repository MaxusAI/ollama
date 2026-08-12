# Draft upstream PR: run the sleep-ordering scheduler tests under testing/synctest

> **DRAFT 2026-08-12** — do not file without an explicit go-ahead. Unlike
> [`upstream-gemma4-sizing-issue.md`](upstream-gemma4-sizing-issue.md), this draft
> needs **no** regeneration: the patch was generated from `upstream/main` at
> `948f6933` (2026-08-11), applies cleanly to it, and was verified by running
> upstream's own suite in a detached worktree. Re-check if upstream touches
> `server/sched_test.go` before filing.

Fork-internal draft for filing against **ollama/ollama**. Patch:
[`patches/upstream-sched-synctest.patch`](patches/upstream-sched-synctest.patch)
(+38/−47 against `server/sched_test.go`, no other file touched).

The fork's equivalent landed as `c6f3fde5`. The fork's own copy of this change
additionally rewords the `schedTestTimeout` helper comment, which does not exist
upstream — that is the only difference between the two, and the six migrated test
bodies are byte-identical between fork and upstream.

The scheduler defect this work surfaced is **not** part of this PR; it is written
up separately in
[`upstream-sched-head-of-line-blocking.md`](upstream-sched-head-of-line-blocking.md).

---

**Title:** server: run the sleep-ordering scheduler tests under `testing/synctest`

## Summary

Six tests in `server/sched_test.go` sequence the scheduler with `time.Sleep` and
depend on orderings they do not enforce. The clearest one is in
`TestSchedRequestsMultipleLoadedModels`:

```go
// Mark b done so it can unload
b.ctxDone()                                   // sched_test.go:481
// Report recovered VRAM usage so scheduler will finish waiting and unload
time.Sleep(1 * time.Millisecond)              // sched_test.go:483
gMu.Lock()
g.FreeMemory = 24 * format.GigaByte
gMu.Unlock()
```

That 1ms sleep has to finish before model `b`'s 10ms expire timer fires. On Linux
and macOS it reliably does. On Windows, whose timer granularity is ~15.6ms
(golang/go#44608, closed without a fix), it reliably does not.

When it inverts, `processPending` retries the load before the test has published
the recovered VRAM, picks `model-c-10g-cpu`, finds `refCount = 1`, so `sched.go:345`
sends no `expiredCh` while `sched.go:351` waits on `unloadedCh` regardless. Nothing
will ever signal for `c`: its request context is the test context. The wait is
unbounded, so the test consumes its entire budget and reports `timeout`.

Changing only that one sleep to 20ms — emulating what Windows' granularity does to
it — reproduces the CI failure deterministically on `main`, 5/5:

```
--- FAIL: TestSchedRequestsMultipleLoadedModels (1.01s)
    sched_test.go:493: timeout
```

Note the elapsed time: 1.01s against a 1s budget. The test does not run slowly, it
runs until the deadline and stops.

Raising the deadline does not help, because the deadline is not what fails — the
test burns whatever budget it is given.

The tests have also been passing partly by luck. Debug output shows
`unload completed runner.name=model-c-10g-cpu` while the runner that actually
unloaded was `model-b-10g-gpu`: `unloadedCh` is a counter rather than a per-runner
handshake, so an unrelated unload satisfies a wait posted for a different runner.

## Change

The six tests that sequence the scheduler with sleeps —
`TestSchedLoad`, `TestSchedRequestsSimpleReloadSameModel`,
`TestSchedRequestsMultipleLoadedModels`, `TestSchedGetRunner`,
`TestSchedPrematureExpired`, `TestSchedAlreadyCanceled` — run inside a
`testing/synctest` bubble. Inside a bubble the clock is fake and `time.Sleep`
advances it only once every goroutine is durably blocked, so "1ms finishes before
10ms" is guaranteed on every platform rather than holding wherever sleeps happen to
be precise.

Because the ordering is now enforced rather than raced, those tests need no wall
clock at all: their `context.WithTimeout` deadlines and all nine
`case <-ctx.Done(): t.Fatal("timeout")` arms are removed. Two of those nine sit
inside the `closeWait` poll loop, which folds into

```go
for !b.srv.closeCalled {
    time.Sleep(1 * time.Millisecond)
}
```

The 28 remaining `context.WithTimeout` sites in the file are untouched. They guard a
channel receive with no ordering dependency and are not affected by this.

Each body moves into its own named function passed to `synctest.Test` rather than
being indented into a closure, so the diff stays confined to the lines that
actually change.

`testing/synctest` is stable as of Go 1.25; `go.mod` is already at `go 1.26.0`, and
CI resolves its toolchain with `go-version-file: 'go.mod'`, so no CI change is
needed.

## A note on what this exposes

A bubble reports a scheduler that parks on a signal nothing will send as an
immediate deadlock naming the blocked line, instead of a platform-specific timeout:

```
panic: deadlock: all goroutines in bubble are blocked
goroutine 66 [select (durable), synctest bubble 1]:
  server.(*Scheduler).processPending   server/sched.go:351
```

That is a diagnostic improvement over a 10-second `timeout` with no stack, and it
is how the `refCount > 0` park above was identified. This PR does not change
`server/sched.go`; the park itself looks like a genuine defect and is worth its own
issue.

## Verification

Against `upstream/main` at `948f6933`, macOS arm64, Go 1.26.5 — the patch was
generated from that tree and every number below was measured in it, not in the fork:

- The failure above reproduces 5/5 on the unpatched tree with the one-line
  perturbation, and the patched tree passes.
- `gofmt` clean; `go vet ./server/` clean. (`gofumpt` reports one pre-existing nit
  at `sched_test.go:2139`, a missing blank line after `mockLlm.Close`, present
  before this change and left alone.)
- Full `server` package green.
- The six migrated tests, 150 iterations each under `-race`, no failures.
- `TestSchedPrematureExpired` drops from ~135ms of real sleeping to 0.01s, since the
  fake clock collapses `time.Sleep(100 * time.Millisecond)`.

Not verified on Windows CI — the change removes the dependency on timer precision
rather than accommodating it, so there is nothing platform-specific left to tune,
but a Windows run on the PR would confirm it.
