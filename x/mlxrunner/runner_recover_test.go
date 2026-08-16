package mlxrunner

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// A panic while evaluating a request must reach the caller as an error. Before
// this, it unwound past the error path in Run and killed the subprocess, so the
// client saw a dead runner and a Go stack instead of a reason.
func TestRunRequestRecoversPanic(t *testing.T) {
	r := &Runner{}
	req := Request{
		Ctx: context.Background(),
		Pipeline: func(context.Context, Request) error {
			panic("mlx: cudaMallocAsync(&data, size, stream) failed: out of memory")
		},
	}

	err := r.runRequest(req)
	if err == nil {
		t.Fatal("expected an error from a panicking pipeline, got nil")
	}

	var fatal fatalRunnerError
	if !errors.As(err, &fatal) {
		t.Errorf("expected fatalRunnerError so Run stops the runner, got %T", err)
	}
	if !strings.Contains(err.Error(), "out of memory") {
		t.Errorf("error should carry the original cause, got %q", err)
	}
	// The one thing a stack trace cannot tell an operator: what to change.
	if !strings.Contains(err.Error(), "num_ctx") {
		t.Errorf("an out-of-memory abort should say what to do, got %q", err)
	}
}

// Non-memory panics must still be reported and still stop the runner, but must
// not be dressed up with advice that does not apply.
func TestRunRequestRecoversNonMemoryPanic(t *testing.T) {
	r := &Runner{}
	req := Request{
		Ctx: context.Background(),
		Pipeline: func(context.Context, Request) error {
			panic("mlx: shape mismatch in matmul")
		},
	}

	err := r.runRequest(req)
	if err == nil {
		t.Fatal("expected an error from a panicking pipeline, got nil")
	}
	var fatal fatalRunnerError
	if !errors.As(err, &fatal) {
		t.Errorf("expected fatalRunnerError, got %T", err)
	}
	if strings.Contains(err.Error(), "num_ctx") {
		t.Errorf("shape mismatch is not an OOM; advice must not be attached: %q", err)
	}
	if !strings.Contains(err.Error(), "shape mismatch") {
		t.Errorf("error should carry the original cause, got %q", err)
	}
}

// An ordinary error must pass through untouched: it is a request-level failure,
// and marking it fatal would tear down a healthy runner.
func TestRunRequestOrdinaryErrorIsNotFatal(t *testing.T) {
	r := &Runner{}
	sentinel := errors.New("prompt too long")
	req := Request{
		Ctx:      context.Background(),
		Pipeline: func(context.Context, Request) error { return sentinel },
	}

	err := r.runRequest(req)
	if !errors.Is(err, sentinel) {
		t.Fatalf("expected the pipeline error to pass through, got %v", err)
	}
	var fatal fatalRunnerError
	if errors.As(err, &fatal) {
		t.Error("an ordinary request error must not stop the runner")
	}
}

// Success must stay success — the deferred recover must not manufacture one.
func TestRunRequestSuccessUnaffected(t *testing.T) {
	r := &Runner{}
	req := Request{
		Ctx:      context.Background(),
		Pipeline: func(context.Context, Request) error { return nil },
	}
	if err := r.runRequest(req); err != nil {
		t.Fatalf("expected nil, got %v", err)
	}
}
