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

// TestSchedAbandonedEvictionDoesNotBlockQueue covers the pending-loop park
// described in docs/maxusai/upstream-sched-head-of-line-blocking.md.
//
// processPending serializes deliberately — it blocks other requests until the
// one in hand is running. The defect is that the park it uses to wait for an
// eviction watches only the scheduler's lifetime context, so a request whose
// client has already hung up goes on holding the whole queue, and it holds it
// against a signal that nothing is obliged to send: when the runner picked for
// eviction is busy, no expiredCh is sent, and the unload only happens if and
// when that runner's in-flight request finishes.
//
//  1. maxRunners=1; load A and hold its request open, so A stays refCount 1.
//  2. Submit B. It needs room, findRunnerToUnload returns busy A,
//     processPending sends no expiredCh (refCount > 0) and parks.
//  3. Cancel B's context: the client behind B gives up.
//  4. Submit A2, a request for the already-loaded A. It needs no eviction and
//     should be served straight from s.loaded.
//
// Under synctest an unreleasable park fails as an immediate bubble deadlock
// naming sched.go's select, rather than as a timeout.
func TestSchedAbandonedEvictionDoesNotBlockQueue(t *testing.T) {
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

		// A2 targets the already-loaded A, so it needs no eviction and must not
		// queue behind B's abandoned wait.
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

// TestSchedAbandonedOOMEvictAllDoesNotBlockQueue covers the second park in the
// same document: evictAllAndWait has the identical defect, on the OOM retry
// path.
//
// It waits for one unloadedCh per runner it expired while watching only the
// scheduler's lifetime context. A runner that is still referenced gets no
// expiredCh — the code only sends one at refCount <= 0 — so its unload happens
// only if and when its in-flight request finishes. Until then the whole pending
// loop is parked, and the client that triggered it can hang up without
// releasing anything.
//
// This could not take the same one-line fix as the pending-loop park: the
// function's bool return already means "scheduler shutting down" and is wired
// to `return` out of processPending, so reporting "the requester gave up"
// needed the outcome to become a third state rather than a second false.
//
//  1. maxRunners=2; load A and hold its request open, so A stays refCount 1.
//  2. Submit B with a loadFn that reports needEvict on an OOM retry, which is
//     the only route into evictAllAndWait. A is busy, so no expiredCh is sent
//     and the wait cannot be satisfied.
//  3. Cancel B's context: the client behind B gives up.
//  4. Submit A2 for the already-loaded A. It needs no eviction at all.
//
// Under synctest an unreleasable park fails as an immediate bubble deadlock
// naming evictAllAndWait's select, rather than as a timeout.
func TestSchedAbandonedOOMEvictAllDoesNotBlockQueue(t *testing.T) {
	t.Setenv("OLLAMA_MAX_LOADED_MODELS", "2")
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

		// Drive B down the OOM retry path. load() sets oomRetryAttempted itself
		// after a post-spawn crash; forcing it here reaches evictAllAndWait
		// without needing to simulate the crash.
		s.newServerFn = b.newServer
		s.loadFn = func(req *LlmRequest, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) bool {
			req.oomRetryAttempted = true
			return true // needEvict
		}
		s.pendingReqCh <- b.req
		synctest.Wait()

		// The client behind B gives up while evictAllAndWait is parked on an
		// unload that busy A will never produce.
		b.ctxDone()
		synctest.Wait()

		// A2 targets the already-loaded A, so it needs no eviction and must not
		// queue behind B's abandoned evict-all.
		s.loadFn = s.load
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
