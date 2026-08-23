package mlxrunner

import (
	"fmt"
	"log/slog"
	"runtime/debug"
)

// firstPanic is what guardClose re-raises once it has seen a panic start
// unwinding a pipeline: the original value, plus the stack captured while the
// frames of the site that panicked were still live beneath the deferred call.
// It implements error so that mlxthread's panicError and recoverRequest, which
// both print the value with %v, lead with the ORIGINAL message.
type firstPanic struct {
	value any
	stack []byte
}

func (p *firstPanic) Error() string {
	return fmt.Sprintf("%v\n\nfirst panic stack:\n%s", p.value, p.stack)
}

// Unwrap lets errors.Is/As reach a panic value that was itself an error.
func (p *firstPanic) Unwrap() error {
	if err, ok := p.value.(error); ok {
		return err
	}
	return nil
}

// guardClose runs a deferred cleanup so that it cannot hide the panic that is
// unwinding the pipeline.
//
// Go's recover only ever returns the MOST RECENT panic. A request pipeline
// defers several cleanups (prefix-cache session, media, speculation, decoder,
// teardown) and each of them touches MLX state; once an evaluation has thrown,
// the first cleanup to reach the abandoned encoder throws again, and THAT is
// the value every recover site downstream sees. The graph-cache thrashing fuse
// was diagnosed for a day as cudaGraphAddDependencies for exactly this reason
// (docs/maxusai/mlx-thrash-check-masks-as-cudagraph.md).
//
// Used as `defer guardClose("name", x.close)`. Because guardClose is itself the
// deferred function, recover inside it returns the panic unwinding the frame,
// if any. On the normal path it is a plain call, and a cleanup that is the
// FIRST thing to fail still propagates as it always did. While unwinding it
// runs the cleanup under its own recover — a second panic is logged, not
// propagated — and re-raises the first, wrapped so the log and the error
// returned to the client name the real cause.
func guardClose(name string, close func()) {
	first := recover()
	if first == nil {
		close()
		return
	}
	wrapped, ok := first.(*firstPanic)
	if !ok {
		wrapped = &firstPanic{value: first, stack: debug.Stack()}
	}
	func() {
		defer func() {
			if second := recover(); second != nil {
				slog.Error("cleanup panicked while the request was already unwinding a panic; keeping the first",
					"cleanup", name, "second", fmt.Sprint(second), "first", fmt.Sprint(wrapped.value))
			}
		}()
		close()
	}()
	panic(wrapped)
}
