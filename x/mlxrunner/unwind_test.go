package mlxrunner

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
)

// recoverValue runs fn and returns what it panicked with, or nil.
func recoverValue(fn func()) (v any) {
	defer func() { v = recover() }()
	fn()
	return nil
}

// A cleanup that panics while the pipeline is already unwinding must not
// replace the cause. Before this, every recover site downstream reported the
// cleanup's failure (cudaGraphAddDependencies) and the real one (the graph
// cache thrashing throw) never reached a log.
func TestGuardCloseKeepsTheFirstPanic(t *testing.T) {
	var order []string
	v := recoverValue(func() {
		defer guardClose("outer", func() { order = append(order, "outer") })
		defer guardClose("inner", func() {
			order = append(order, "inner")
			panic("mlx: cudaGraphAddDependencies(...) failed: invalid argument")
		})
		panic("mlx: Graph cache thrashing detected")
	})
	fp, ok := v.(*firstPanic)
	if !ok {
		t.Fatalf("expected *firstPanic, got %T: %v", v, v)
	}
	if fp.value != "mlx: Graph cache thrashing detected" {
		t.Errorf("first panic value lost: %v", fp.value)
	}
	msg := fp.Error()
	if !strings.HasPrefix(msg, "mlx: Graph cache thrashing detected") {
		t.Errorf("Error() must lead with the first cause, got %q", msg)
	}
	if !strings.Contains(msg, "first panic stack") || !strings.Contains(msg, "TestGuardCloseKeepsTheFirstPanic") {
		t.Errorf("Error() should carry the stack of the site that panicked first: %q", msg)
	}
	if strings.Contains(msg, "cudaGraphAddDependencies") {
		t.Errorf("the second panic must not be promoted into the reported cause: %q", msg)
	}
	if got := strings.Join(order, ","); got != "inner,outer" {
		t.Errorf("every cleanup must still run, in defer order; ran %q", got)
	}
}

// On the normal path the guard is a plain call.
func TestGuardCloseRunsCleanupWithoutPanic(t *testing.T) {
	ran := false
	v := recoverValue(func() {
		defer guardClose("plain", func() { ran = true })
	})
	if v != nil {
		t.Fatalf("unexpected panic %v", v)
	}
	if !ran {
		t.Error("cleanup did not run")
	}
}

// Only a SECOND panic is suppressed. A cleanup that is the first thing to fail
// still propagates exactly as a bare deferred call would.
func TestGuardCloseLetsAFirstCleanupPanicThrough(t *testing.T) {
	v := recoverValue(func() {
		defer guardClose("cleanup", func() { panic("close failed on its own") })
	})
	if v == nil || !strings.Contains(fmt.Sprint(v), "close failed on its own") {
		t.Fatalf("a cleanup's own first panic must propagate, got %v", v)
	}
	if _, wrapped := v.(*firstPanic); wrapped {
		t.Error("a first panic raised by a cleanup is not a masked panic and must not be re-wrapped")
	}
}

// Through the runner's recover path the client and the log see the original
// cause first, with the runner's usual prefix.
func TestRunRequestReportsFirstPanicThroughGuard(t *testing.T) {
	r := &Runner{}
	req := Request{
		Ctx: context.Background(),
		Pipeline: func(context.Context, Request) error {
			defer guardClose("prefix-cache session", func() {
				panic("mlx: cudaGraphAddDependencies(...) failed: invalid argument")
			})
			panic("mlx: Graph cache thrashing detected")
		},
	}
	err := r.runRequest(req)
	if err == nil {
		t.Fatal("expected an error from a panicking pipeline, got nil")
	}
	if !strings.HasPrefix(err.Error(), "mlx runner aborted: mlx: Graph cache thrashing detected") {
		t.Errorf("the reported cause must be the first panic, got %q", err.Error())
	}
}

// Production never takes runRequest's mlxThread == nil branch (server.go always
// builds the Runner with a worker), so the first-panic guarantee has to hold
// through mlxthread too: the worker recovers the *firstPanic, wraps it in its
// own panicError and re-raises it on the calling goroutine, and recoverRequest
// prints that with %v. The message must still lead with the original cause,
// not with the worker-stack preamble.
func TestRunRequestReportsFirstPanicThroughMLXThread(t *testing.T) {
	worker, err := mlxthread.Start("test", func() error { return nil })
	if err != nil {
		t.Fatalf("start worker: %v", err)
	}
	t.Cleanup(func() { _ = worker.Stop(context.Background(), nil) })

	r := &Runner{mlxThread: worker}
	req := Request{
		Ctx: context.Background(),
		Pipeline: func(context.Context, Request) error {
			defer guardClose("prefix-cache session", func() {
				panic("mlx: cudaGraphAddDependencies(...) failed: invalid argument")
			})
			panic("mlx: Graph cache thrashing detected")
		},
	}
	err = r.runRequest(req)
	if err == nil {
		t.Fatal("expected an error from a panicking pipeline, got nil")
	}
	var fatal fatalRunnerError
	if !errors.As(err, &fatal) {
		t.Errorf("expected fatalRunnerError so Run stops the runner, got %T", err)
	}
	if !strings.HasPrefix(err.Error(), "mlx runner aborted: mlx: Graph cache thrashing detected") {
		t.Errorf("through mlxthread the reported cause must still be the first panic, got %q", err.Error())
	}
	if !strings.Contains(err.Error(), "mlx worker stack") {
		t.Errorf("mlxthread's worker stack should still be attached after the cause, got %q", err.Error())
	}
}
