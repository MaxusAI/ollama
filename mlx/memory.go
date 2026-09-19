package mlx

// #include "generated.h"
// #include <stdlib.h>
import "C"

import (
	"fmt"
	"log/slog"
	"strconv"
	"unsafe"
)

func (b Byte) String() string {
	return strconv.FormatInt(int64(b), 10) + " B"
}

func (b KibiByte) String() string {
	return strconv.FormatFloat(float64(b)/(1<<10), 'f', 2, 64) + " KiB"
}

func (b MebiByte) String() string {
	return strconv.FormatFloat(float64(b)/(1<<(2*10)), 'f', 2, 64) + " MiB"
}

func (b GibiByte) String() string {
	return strconv.FormatFloat(float64(b)/(1<<(3*10)), 'f', 2, 64) + " GiB"
}

func (b TebiByte) String() string {
	return strconv.FormatFloat(float64(b)/(1<<(4*10)), 'f', 2, 64) + " TiB"
}

func PrettyBytes(n int) fmt.Stringer {
	switch {
	case n < 1<<10:
		return Byte(n)
	case n < 1<<(2*10):
		return KibiByte(n)
	case n < 1<<(3*10):
		return MebiByte(n)
	case n < 1<<(4*10):
		return GibiByte(n)
	default:
		return TebiByte(n)
	}
}

func ActiveMemory() int {
	var active C.size_t
	mlxCheck(C.mlx_get_active_memory(&active))
	return int(active)
}

func CacheMemory() int {
	var cache C.size_t
	mlxCheck(C.mlx_get_cache_memory(&cache))
	return int(cache)
}

func PeakMemory() int {
	var peak C.size_t
	mlxCheck(C.mlx_get_peak_memory(&peak))
	return int(peak)
}

func ResetPeakMemory() {
	mlxCheck(C.mlx_reset_peak_memory())
}

// MaxRecommendedWorkingSetSize returns the device's recommended upper bound
// for resident Metal allocations.
func MaxRecommendedWorkingSetSize() (int, error) {
	info := mlxCheck(C.mlx_device_info_new())
	if err := mlxError(C.mlx_device_info_get(&info, DefaultDevice().ctx)); err != nil {
		return 0, err
	}
	defer freeDeviceInfo(info)

	key := C.CString("max_recommended_working_set_size")
	defer C.free(unsafe.Pointer(key))

	var size C.size_t
	rc := C.mlx_device_info_get_size(&size, info, key)
	if err := lastError(); err != nil {
		return 0, err
	}
	if rc != 0 {
		// mlx-c reports a missing key with a non-zero return and no message.
		return 0, fmt.Errorf("mlx: no max_recommended_working_set_size in device info")
	}
	return int(size), nil
}

// SetWiredLimit sets the maximum amount of Metal memory MLX keeps resident and
// returns the previous limit.
func SetWiredLimit(limit int) (int, error) {
	if limit < 0 {
		return 0, fmt.Errorf("mlx: wired limit must be non-negative")
	}

	var previous C.size_t
	if err := mlxError(C.mlx_set_wired_limit(&previous, C.size_t(limit))); err != nil {
		return 0, err
	}
	return int(previous), nil
}

// MemoryLimit returns MLX's current allocator limit. Unlike the wired limit
// this is backend-independent: on CUDA it is the only cap available, since
// "wired" residency and the max-recommended-working-set device key are both
// Metal concepts.
func MemoryLimit() (int, error) {
	var size C.size_t
	if err := mlxError(C.mlx_get_memory_limit(&size)); err != nil {
		return 0, err
	}
	return int(size), nil
}

// SetMemoryLimit sets MLX's allocator limit and returns the previous one.
// Allocations beyond it fail rather than being served, so the caller gets an
// error instead of the backend aborting the process.
func SetMemoryLimit(limit int) (int, error) {
	if limit < 0 {
		return 0, fmt.Errorf("mlx: memory limit must be non-negative")
	}

	var previous C.size_t
	if err := mlxError(C.mlx_set_memory_limit(&previous, C.size_t(limit))); err != nil {
		return 0, err
	}
	return int(previous), nil
}

// SetCacheLimit caps the buffer cache — the freed blocks MLX RETAINS for reuse
// rather than returning to the driver — and returns the previous limit.
//
// THIS IS NOT SetMemoryLimit. That one caps TOTAL allocation and decides when an
// allocation fails; it does not make MLX release anything. This one decides how
// much MLX keeps after it is done with it, which is what governs steady-state
// footprint. Measured on gemma4:31b-nvfp4, one 3072x1728 image, CUDA:
//
//	memory.active 18.29 GiB   (live tensors, ~= the weights)
//	memory.cache  13.16 GiB   (retained, and the whole difference)
//	nvidia-smi    32.36 GiB   vs 23.03 GiB for the same work on llama.cpp
//
// active + cache accounts for 31.45 of the 32.36 GiB observed, so the excess is
// MLX's cache and not, as was first assumed, CUDA's async memory pool — which
// would have needed cudaMemPoolTrimTo and been outside our reach.
//
// ClearCache() empties it at a moment in time; this bounds how far it refills.
// Both are needed: the cache regrew to 13 GiB DURING a single prefill, between
// two of the ClearCache calls the pipeline already makes.
//
// Zero is legal and means "retain nothing", which trades allocator churn for the
// smallest footprint.
func SetCacheLimit(limit int) (int, error) {
	if limit < 0 {
		return 0, fmt.Errorf("mlx: cache limit must be non-negative")
	}

	var previous C.size_t
	if err := mlxError(C.mlx_set_cache_limit(&previous, C.size_t(limit))); err != nil {
		return 0, err
	}
	return int(previous), nil
}

type Memory struct{}

func (Memory) LogValue() slog.Value {
	return slog.GroupValue(
		slog.Any("active", PrettyBytes(ActiveMemory())),
		slog.Any("cache", PrettyBytes(CacheMemory())),
		slog.Any("peak", PrettyBytes(PeakMemory())),
	)
}

type (
	Byte     int
	KibiByte int
	MebiByte int
	GibiByte int
	TebiByte int
)

func ClearCache() {
	mlxCheck(C.mlx_clear_cache())
}
