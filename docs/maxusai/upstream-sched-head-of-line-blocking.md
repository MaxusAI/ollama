# Draft upstream issue: processPending parks the whole queue on unloadedCh after picking a busy runner

> **DRAFT 2026-08-12** — do not file without an explicit go-ahead. Verified
> against `upstream/main` at `948f6933` (2026-08-11): `server/sched.go` differs
> from the fork by 8 added lines elsewhere in the file, and every site below is
> byte-identical upstream. Line numbers in this document are **upstream's**;
> they coincide with the fork's except `findRunnerToUnload` (fork 1698, upstream
> 1690).

Fork-internal draft for filing against **ollama/ollama**. Issues are disabled on
MaxusAI/ollama, so this doc is the fork's record of the finding, following the
precedent of [`upstream-gemma4-sizing-issue.md`](upstream-gemma4-sizing-issue.md).

**Fork status: both parks fixed; the serialization below is not.** Everything
described in the issue text still stands upstream, which is what that text
describes. The fork now carries the scoped arm described
at the end of *Suggested fixes* — a `case <-pending.ctx.Done():` on the select at
`sched.go:351`, which breaks out of the per-request scheduling loop instead of
holding it. That closes the unbounded case: a request whose client has hung up no
longer holds the queue against a signal nothing is obliged to send. It is the
smallest change that fixes the reproducer below, and it is regression-tested by
`TestSchedAbandonedEvictionDoesNotBlockQueue` in `server/sched_headofline_test.go`.

**Still present, deliberately:** while a *live* client waits for eviction, the
loop is still serialized, so requests that need no eviction still queue behind it
for as long as the picked runner's in-flight request runs. Removing that means
either teaching `findRunnerToUnload` to report whether it picked an idle runner,
or restructuring the loop to set a blocked request aside and keep dequeuing —
both are behaviour changes to untouched upstream code whose correct shape depends
on upstream intent, and both risk carrying live scheduler divergence.

**`evictAllAndWait` is now fixed too.** It had the identical park — waiting on
`unloadedCh` while watching only the scheduler's lifetime context, reachable on
the OOM post-crash retry path. It could not take the same one-line arm, because
its `bool` return had already spent its only `false` on "scheduler shutting
down", wired to `return` out of `processPending`. The outcome is now a
three-state `evictOutcome` (`evictComplete` / `evictShutdown` /
`evictAbandoned`), so an abandoned requester releases the loop while a shutdown
still takes it down, and the wait watches `pending.ctx` alongside `ctx`.
Regression-tested by `TestSchedAbandonedOOMEvictAllDoesNotBlockQueue`, which
deadlocks at `evictAllAndWait`'s select without the change.

---

**Title:** sched: `processPending` parks the whole queue on `unloadedCh` after
picking a busy runner, blocking requests that need no eviction

## Summary

`processPending` can park the entire scheduler queue on a signal that nothing will
ever send. While it is parked, requests that need **no eviction at all** — including
requests for a model that is already loaded — are never dequeued, and the request
that caused the park cannot release it by being cancelled.

Three sites combine.

**1. `findRunnerToUnload` returns a busy runner.** It scans for an idle runner
first, and when every runner is busy it falls through and returns `runnerList[0]`
regardless of `refCount` (`sched.go:1690-1692`):

```go
// None appear idle, just wait for the one with the shortest duration
slog.Debug("no idle runners, picking the shortest duration", "runner_count", len(runnerList), "runner", runnerList[0])
return runnerList[0]
```

**2. The caller requests nothing, then waits anyway** (`sched.go:344-358`):

```go
runnerToExpire.sessionDuration = 0
if runnerToExpire.refCount <= 0 {
    s.expiredCh <- runnerToExpire        // not taken when the runner is busy
}
runnerToExpire.refMu.Unlock()
// Wait for the unload to happen
select {
case <-ctx.Done():                       // scheduler lifetime ctx, never pending.ctx
    return
case <-s.unloadedCh:
    continue
}
```

The select watches only the scheduler's lifetime context. `pending.ctx` is checked
once at dequeue (`sched.go:237`) and never again, so a client hanging up does not
release the loop.

**3. `unloadedCh` is a counter, not a per-runner handshake.** Any runner's unload
satisfies any waiter, and the idle branch discards signals outright
(`sched.go:360-362`):

```go
case <-s.unloadedCh:
    // An unload request when there are no pending request can be ignored
    slog.Debug("ignoring unload event with no pending requests")
```

## Minimal reproduction

Deterministic under `testing/synctest` — no wall clock involved, fails in 0.01s.
Drop into `server/` as `sched_probe_test.go`:

```go
package server

import (
	"context"
	"testing"
	"testing/synctest"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
)

// TestProbePendingLoopHeadOfLineBlocking asks whether the unloadedCh park in
// processPending blocks requests that need no eviction at all.
//
//  1. maxRunners=1; load A and hold its request open, so A stays refCount 1.
//  2. Submit B. It needs room, findRunnerToUnload returns busy A, processPending
//     sends no expiredCh (refCount > 0) and parks on unloadedCh.
//  3. Cancel B's context: the client behind B gives up.
//  4. Submit A2, a request for the already-loaded A. It needs no eviction and
//     should be served straight from s.loaded.
//
// If A2 is never served, the park is head-of-line blocking, not backpressure.
func TestProbePendingLoopHeadOfLineBlocking(t *testing.T) {
	t.Setenv("OLLAMA_MAX_LOADED_MODELS", "1")
	synctest.Test(t, func(t *testing.T) {
		ctx := t.Context()
		s := InitScheduler(ctx)
		s.waitForRecovery = 10 * time.Millisecond
		g := ml.DeviceInfo{DeviceID: ml.DeviceID{Library: "Metal"}}
		g.TotalMemory = 24 * format.GigaByte
		g.FreeMemory = 24 * format.GigaByte
		s.getGpuFn = func(ctx context.Context, runners []ml.FilteredRunnerDiscovery) []ml.DeviceInfo {
			return []ml.DeviceInfo{g}
		}
		s.getSystemInfoFn = getSystemInfoFn

		gpu := map[ml.DeviceID]uint64{{Library: "Metal"}: 1 * format.GigaByte}
		a := newScenarioRequest(t, ctx, "model-a", 1*format.GigaByte, nil, gpu)
		a.req.sessionDuration = &api.Duration{Duration: time.Hour}
		b := newScenarioRequest(t, ctx, "model-b", 1*format.GigaByte, nil, gpu)

		s.newServerFn = a.newServer
		s.pendingReqCh <- a.req
		s.Run(ctx)
		select {
		case <-a.req.successCh:
		case err := <-a.req.errCh:
			t.Fatal(err.Error())
		}
		// A is loaded and still referenced: a.ctxDone() is never called.

		// B needs room, and the only runner is busy A.
		s.newServerFn = b.newServer
		s.pendingReqCh <- b.req
		synctest.Wait()

		// The client behind B gives up.
		b.ctxDone()
		synctest.Wait()

		// A2 targets the already-loaded A, so it needs no eviction.
		a2 := newScenarioRequest(t, ctx, "model-a", 1*format.GigaByte, nil, gpu)
		a2.req.model.ModelPath = a.req.model.ModelPath
		s.pendingReqCh <- a2.req

		select {
		case <-a2.req.successCh:
		case err := <-a2.req.errCh:
			t.Fatal(err.Error())
		}
	})
}
```

Result:

```
--- FAIL: TestProbePendingLoopHeadOfLineBlocking (0.01s)
panic: deadlock: all goroutines in bubble are blocked

goroutine 68 [select (durable), synctest bubble 1]:
  server.(*Scheduler).processPending   server/sched.go:351
goroutine 52 [chan receive (durable), synctest bubble 1]:
  ...                                  server/sched_probe_test.go:68
```

`processPending` is parked at `sched.go:351`; the test goroutine is blocked
receiving on `a2.req.successCh` — a request for a model that is already loaded.

## Impact

On a server where every loaded runner is busy, a single request for a
not-yet-loaded model freezes **all** scheduling until one of the in-flight
requests completes. Requests that would hit an already-loaded runner queue behind
it. The client that triggered the park can hang up and nothing changes.

Reachability needs all loaded runners busy at once plus a new model that needs
room — plausible on a busy multi-model host, not an everyday state. The stall
lasts as long as the in-flight request on the runner that got picked, so a long
generation means a long stall.

## How it surfaced

This is the mechanism behind intermittent `TestSchedRequestsMultipleLoadedModels`
timeouts on `windows-latest`. Upstream's test sequences the scheduler with
`time.Sleep` and encodes an ordering it does not enforce: at
`sched_test.go:481-483` it calls `b.ctxDone()` then sleeps 1ms, and needs that
sleep to finish before model `b`'s 10ms expire timer fires. Windows' ~15.6ms timer
granularity (golang/go#44608) inverts it. `processPending` then retries the load
before the test has published the recovered VRAM, picks `model-c-10g-cpu`, finds
`refCount=1`, sends no `expiredCh`, and parks — so the test burns its entire
budget and reports `timeout`.

Changing only that one sleep to 20ms reproduces the CI failure deterministically,
5/5, with the identical signature.

The test previously passed by luck. The debug log shows
`unload completed runner.name=model-c-10g-cpu` while the runner that actually
unloaded was `model-b-10g-gpu` — an unrelated unload satisfying a wait posted for
`c`, which is defect 3 above observable in upstream's own test output.

## Suggested fixes

Roughly increasing blast radius:

- add `case <-pending.ctx.Done():` to the select at `sched.go:351`, so an
  abandoned request stops holding the queue
- have `findRunnerToUnload` return `(runner, isIdle)` so the caller can skip the
  wait when it requested no unload
- make `unloadedCh` carry the unloaded runner, so a waiter can tell its own unload
  from someone else's

## Related

The test-side half is a separate, self-contained change — see
[`upstream-sched-synctest-pr.md`](upstream-sched-synctest-pr.md). It makes those
orderings deterministic and turns a park of this shape into an immediate deadlock
report naming the blocked line, instead of a platform-specific timeout. It does
not fix this defect. Landed in the fork as `c6f3fde5`.
