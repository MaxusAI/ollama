package mlx

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"
)

func TestSetWiredLimitRejectsOversizeWithoutChangingLimit(t *testing.T) {
	skipIfNoMLX(t)
	if !GPUIsAvailable() {
		t.Skip("MLX GPU not available")
	}

	var testErr error
	withMLXThread(t, func() {
		testErr = checkWiredLimitRejectsOversize()
	})
	if testErr != nil {
		t.Fatal(testErr)
	}
}

func checkWiredLimitRejectsOversize() (err error) {
	maxRecommended, err := MaxRecommendedWorkingSetSize()
	if err != nil {
		return err
	}
	if maxRecommended == math.MaxInt {
		return errors.New("recommended working set cannot be exceeded by an int")
	}

	previous, err := SetWiredLimit(maxRecommended)
	if err != nil {
		return err
	}
	defer func() {
		if _, restoreErr := SetWiredLimit(previous); restoreErr != nil {
			err = errors.Join(err, fmt.Errorf("restore wired limit: %w", restoreErr))
		}
	}()

	if _, err := SetWiredLimit(maxRecommended + 1); err == nil {
		return errors.New("SetWiredLimit accepted a limit above the recommended working set")
	}

	current, err := SetWiredLimit(maxRecommended)
	if err != nil {
		return err
	}
	if current != maxRecommended {
		return fmt.Errorf("wired limit after rejected update = %d, want %d", current, maxRecommended)
	}

	a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	b := Matmul(a, a)
	Eval(b)
	if got, want := b.Floats(), []float32{7, 10, 15, 22}; !slices.Equal(got, want) {
		return fmt.Errorf("evaluation after rejected update = %v, want %v", got, want)
	}
	return nil
}

// SetCacheLimit must bound the RETAINED CACHE and nothing else. The distinction
// this asserts is the one memory.go's doc comment is written to defend, and it
// is undefendable by review alone: mlx_set_cache_limit and mlx_set_memory_limit
// have identical Go-side signatures, so swapping one for the other compiles,
// keeps every existing test green, and silently converts a cache bound into a
// cap on total allocation.
//
// A round-trip on SetCacheLimit alone cannot catch that -- both C calls return
// the previous value and would round-trip identically. Cross-checking the
// memory limit does: it must be untouched.
func TestSetCacheLimitLeavesTheMemoryLimitAlone(t *testing.T) {
	skipIfNoMLX(t)
	if !GPUIsAvailable() {
		t.Skip("MLX GPU not available")
	}

	var testErr error
	withMLXThread(t, func() {
		testErr = checkCacheLimitDoesNotTouchMemoryLimit()
	})
	if testErr != nil {
		t.Fatal(testErr)
	}
}

func checkCacheLimitDoesNotTouchMemoryLimit() (err error) {
	before, err := MemoryLimit()
	if err != nil {
		return err
	}

	previous, err := SetCacheLimit(1 << 20)
	if err != nil {
		return err
	}
	defer func() {
		if _, restoreErr := SetCacheLimit(previous); restoreErr != nil && err == nil {
			err = fmt.Errorf("restoring cache limit %d: %w", previous, restoreErr)
		}
	}()

	after, err := MemoryLimit()
	if err != nil {
		return err
	}
	if after != before {
		return fmt.Errorf("SetCacheLimit moved the MEMORY limit %d -> %d; it must bound the retained cache only", before, after)
	}
	return nil
}

// The guard is unreachable through parseCacheLimit, which cannot produce a
// negative, but SetCacheLimit is exported and the C call takes a size_t -- a
// negative would wrap to an enormous unsigned value and silently uncap the
// cache. Cheap to pin, and it needs no GPU.
func TestSetCacheLimitRejectsNegative(t *testing.T) {
	if _, err := SetCacheLimit(-1); err == nil {
		t.Fatal("expected a negative cache limit to be refused before reaching the C call")
	}
}
