package mlx

import (
	"context"
	"runtime"
	"sync"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
)

func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
	// Each test runs on its own goroutine, so each one that drives MLX directly
	// has to take ownership of an OS thread before its first operation.
	ClaimOSThread()
}

func startMLXThread(t *testing.T) *mlxthread.Thread {
	t.Helper()

	thread, err := mlxthread.Start("mlx-test", func() error {
		if err := CheckInit(); err != nil {
			return err
		}
		ClaimOSThread()
		if GPUIsAvailable() {
			SetDefaultDeviceGPU()
		}
		return nil
	})
	if err != nil {
		t.Skipf("MLX not available: %v", err)
	}

	return thread
}

func stopMLXThread(t *testing.T, thread *mlxthread.Thread) {
	t.Helper()

	if err := thread.Stop(context.Background(), func() {
		Sweep()
		ClearCache()
		resetDefaultStreamCache()
	}); err != nil {
		t.Fatal(err)
	}
}

func withMLXThread(t *testing.T, fn func()) {
	t.Helper()

	thread := startMLXThread(t)
	defer stopMLXThread(t, thread)

	if err := thread.Do(context.Background(), func() error {
		fn()
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// TestMLXOperationsSurviveRescheduling drives MLX from a plain test goroutine
// while every P is busy and the goroutine yields between operations, which is
// what makes the Go scheduler re-dispatch a goroutine onto a different M.
//
// MLX resolves its default stream from thread-local storage and keeps the Metal
// command encoder behind it in a thread_local map, so an array built on one OS
// thread cannot be evaluated on another. Without a thread claim that spans the
// whole sequence, Eval here fails with "There is no Stream(gpu, 0) in current
// thread" — reliably under this much scheduler pressure, and intermittently in
// ordinary test runs.
func TestMLXOperationsSurviveRescheduling(t *testing.T) {
	skipIfNoMLX(t)

	oldProcs := runtime.GOMAXPROCS(8)
	defer runtime.GOMAXPROCS(oldProcs)

	// Saturate every P so a yielding goroutine is very likely to come back on a
	// different thread.
	stop := make(chan struct{})
	var busy sync.WaitGroup
	for range 16 {
		busy.Add(1)
		go func() {
			defer busy.Done()
			for {
				select {
				case <-stop:
					return
				default:
					runtime.Gosched()
				}
			}
		}()
	}
	defer func() {
		close(stop)
		busy.Wait()
	}()

	for i := range 200 {
		a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
		runtime.Gosched()
		b := Matmul(a, a)
		runtime.Gosched()
		Eval(b)
		if got := b.Floats(); len(got) != 4 {
			t.Fatalf("iteration %d: got %d values, want 4", i, len(got))
		}
		Sweep()
		if i%16 == 0 {
			runtime.GC()
		}
	}
}

func TestThreadedMLXOperations(t *testing.T) {
	thread := startMLXThread(t)
	defer stopMLXThread(t, thread)

	oldProcs := runtime.GOMAXPROCS(8)
	defer runtime.GOMAXPROCS(oldProcs)

	const goroutines = 8
	const iterations = 8

	var wg sync.WaitGroup
	errCh := make(chan error, goroutines)
	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()

			for range iterations {
				if err := thread.Do(context.Background(), func() error {
					a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
					b := Matmul(a, a)
					AsyncEval(b)
					Eval(b)
					Sweep()
					ClearCache()
					return nil
				}); err != nil {
					errCh <- err
					return
				}
			}
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatal(err)
	}
}
