package mlx

//go:generate go run generator/main.go -output=. ./include/mlx/c/*.h

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/include
// #cgo LDFLAGS: -lstdc++
// #cgo darwin LDFLAGS: -framework Foundation -framework Metal -framework Accelerate
// #include "generated.h"
// #include <string.h>
//
// static __thread char _mlx_last_error_msg[1024] = {0};
// static __thread int  _mlx_last_error_flag = 0;
// static __thread int  _mlx_thread_owned = 0;
//
// // _mlx_thread_owned marks an OS thread that a goroutine has claimed for MLX.
// // Go has no goroutine-local storage, but the Go runtime only ever schedules
// // a locked goroutine on its own thread, so "this thread is claimed" is
// // equivalent to "this goroutine already claimed it".
// static int mlx_thread_owned(void) {
//     return _mlx_thread_owned;
// }
//
// static void mlx_thread_take_ownership(void) {
//     _mlx_thread_owned = 1;
// }
//
// static void _mlx_capture_error_handler(const char* msg, void* data) {
//     (void)data;
//     strncpy(_mlx_last_error_msg, msg, sizeof(_mlx_last_error_msg) - 1);
//     _mlx_last_error_msg[sizeof(_mlx_last_error_msg) - 1] = '\0';
//     _mlx_last_error_flag = 1;
// }
//
// static void mlx_install_capture_handler(void) {
//     if (mlx_set_error_handler_) {
//         mlx_set_error_handler_(_mlx_capture_error_handler, NULL, NULL);
//     }
// }
//
// static void mlx_clear_last_error(void) {
//     _mlx_last_error_flag = 0;
//     _mlx_last_error_msg[0] = '\0';
// }
//
// static const char* mlx_get_last_error(void) {
//     return _mlx_last_error_flag ? _mlx_last_error_msg : "";
// }
import "C"

import (
	"fmt"
	"runtime"
)

func init() {
	// Replace the default exit(-1) error handler with one that captures
	// the error message so we can surface it in Go.
	C.mlx_install_capture_handler()
}

// Version returns the MLX core library version string.
func Version() string {
	str := C.mlx_string_new()
	defer C.mlx_string_free(str)
	C.mlx_version(&str)
	return C.GoString(C.mlx_string_data(str))
}

// ClaimOSThread binds the calling goroutine to its current OS thread for the
// rest of the goroutine's life and makes that thread this process's MLX owner.
// Call it once, during setup, from the goroutine that will drive MLX — before
// its first MLX operation.
//
// MLX keeps its default streams in thread-local storage (mlx/stream.cpp) and
// the Metal command encoders backing them in a thread_local map
// (mlx/backend/metal/device.cpp), and an array records the stream it was built
// on. That state spans calls, so pinning around a single call is not enough: an
// unpinned goroutine can be rescheduled onto a different OS thread between two
// MLX operations, and evaluating there fails with
// "There is no Stream(gpu, 0) in current thread."
//
// The lock is deliberately never released — the thread-local state outlives any
// single call, so the thread belongs to this goroutine until it exits. Repeated
// calls are no-ops, so the runtime's LockOSThread nesting counter cannot run
// away on a long-lived worker.
func ClaimOSThread() {
	if C.mlx_thread_owned() != 0 {
		return
	}

	// Order matters: the goroutine may still migrate between the check above
	// and the lock, so mark the thread only once it can no longer change.
	runtime.LockOSThread()
	C.mlx_thread_take_ownership()

	// The cached stream belongs to whichever thread resolved it, so a new owner
	// must resolve its own.
	resetDefaultStreamCache()
}

// mlxCall claims the OS thread so the thread-local error state is read from the
// same thread that executed fn, and so the goroutine cannot migrate away from
// the streams its arrays were built on.
func mlxCall(fallback string, fn func() C.int) error {
	ClaimOSThread()

	C.mlx_clear_last_error()
	if fn() != 0 {
		msg := C.GoString(C.mlx_get_last_error())
		if msg == "" {
			msg = fallback
		}
		return fmt.Errorf("mlx: %s", msg)
	}
	return nil
}

// mlxCheck panics with the captured MLX error. Most array operations cannot
// recover from a failed graph construction or evaluation.
func mlxCheck(fallback string, fn func() C.int) {
	if err := mlxCall(fallback, fn); err != nil {
		panic(err.Error())
	}
}

func doEval(outputs []*Array, async bool) {
	if len(outputs) == 0 {
		return
	}

	vector := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vector)

	for _, output := range outputs {
		if output != nil && output.Valid() {
			C.mlx_vector_array_append_value(vector, output.ctx)
		}
	}

	mlxCheck("eval failed", func() C.int {
		if async {
			return C.mlx_async_eval(vector)
		}
		return C.mlx_eval(vector)
	})
}

func AsyncEval(outputs ...*Array) {
	doEval(outputs, true)
}

func Eval(outputs ...*Array) {
	doEval(outputs, false)
}

// MetalIsAvailable returns true if a Metal GPU is available.
func MetalIsAvailable() bool {
	var available C._Bool
	C.mlx_metal_is_available(&available)
	return bool(available)
}

// CUDAIsAvailable returns true if a CUDA GPU is available.
func CUDAIsAvailable() bool {
	var available C._Bool
	C.mlx_cuda_is_available(&available)
	return bool(available)
}
