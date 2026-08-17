// Package mlxtest provides shared scaffolding for tests that exercise MLX
// through the cgo wrapper in x/mlxrunner/mlx.
package mlxtest

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// SkipIfUnavailable skips the test when the MLX dynamic library cannot be
// loaded (e.g. no MLX backend built for this platform).
func SkipIfUnavailable(t testing.TB) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

// Setup prepares a test that calls into MLX natively: it skips when MLX is
// unavailable and pins the test goroutine to its OS thread for the duration
// of the test.
//
// The thread pin is load-bearing, not defensive: MLX's default stream cache
// is thread-local, and anything that migrates the goroutine mid-test (the
// race detector's scheduler in particular) otherwise panics with
// "There is no Stream(gpu, 0) in current thread".
//
// Setup deliberately does not switch devices or sweep caches: switching the
// default device re-creates the process-wide default stream, and sweeping the
// allocator cache between tests changes allocator reuse — both perturbed
// tests that share lazy arrays with subtests running on other threads.
//
// The claim is permanent — no paired UnlockOSThread — which is load-bearing in
// the same way the pin is (ADR 0017, and ADR 0018 for what it costs to get
// wrong). Releasing it returns a thread to the runtime's pool with MLX's
// thread-local stream and command-encoder state still on it, so the next
// goroutine scheduled there inherits a stream it cannot evaluate on. That is
// what made x/imagegen segfault: every test passed alone and any two in
// sequence crashed. mlx.ClaimOSThread is idempotent, so calling it once per
// test cannot run the runtime's lock counter away across a package.
func Setup(t testing.TB) {
	t.Helper()

	SkipIfUnavailable(t)

	mlx.ClaimOSThread()
}
