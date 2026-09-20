package mlxrunner

import (
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
)

// An operator-set limit must be honoured. Before this, setEnv replaced it with
// a value derived from free VRAM and the knob did nothing at all -- which also
// made the footprint/throughput question unmeasurable without editing Go.
func TestBudgetOverrideLowersTheCeiling(t *testing.T) {
	const available = 82 << 30
	got, overridden := budgetWithOverride(available, strconv.Itoa(24<<30))
	if !overridden {
		t.Fatal("expected the environment value to be reported as an override")
	}
	if got != 24<<30 {
		t.Errorf("got %d, want the requested 24GiB", got)
	}
}

// It may only ask for LESS. Honouring a larger request would reintroduce the
// overcommit that deriving from FREE memory was meant to prevent.
func TestBudgetOverrideCannotRaiseTheCeiling(t *testing.T) {
	const available = 32 << 30
	got, overridden := budgetWithOverride(available, strconv.Itoa(64<<30))
	if got != available {
		t.Errorf("got %d, want it clamped to available %d", got, available)
	}
	if !overridden {
		t.Error("a clamped request is still an override and must be logged as one")
	}
}

// Unset, empty and unparseable all mean "no override": behaviour must be
// byte-identical to before this existed.
func TestBudgetOverrideAbsentIsInert(t *testing.T) {
	const available = 82 << 30
	for _, env := range []string{"", "0", "not-a-number", "-5", "18446744073709551615999"} {
		got, overridden := budgetWithOverride(available, env)
		if overridden {
			t.Errorf("env %q must not count as an override", env)
		}
		if got != available {
			t.Errorf("env %q: got %d, want the derived %d", env, got, available)
		}
	}
}

// The three tests above exercise budgetWithOverride directly. These exercise
// Client.Load, because the unit tests pass even if the call site is deleted --
// the override could be entirely unwired and the suite would stay green.

// An operator cap BELOW the model size must refuse at Load. Before this it was
// admitted: `available` decided admission while `vramBudget` was what the runner
// actually got, and the two stopped being the same number when the override
// landed. Measured on the numbers from the override PR's own benchmark --
// gemma4:31b-nvfp4 at 18.29 GiB, 88 GiB free, capped at 8 GiB -- Load returned
// nil and handed the subprocess a ceiling 10.3 GiB under the weights.
func TestLoadRefusesAnOverrideBelowTheModelSize(t *testing.T) {
	t.Setenv(MemoryLimitEnv, strconv.Itoa(8<<30))

	var c Client
	c.memory.Store(19_634_601_000) // 18.29 GiB
	gpus := []ml.DeviceInfo{{FreeMemory: 88 << 30}}

	_, err := c.Load(t.Context(), ml.SystemInfo{}, gpus, false)
	if err == nil {
		t.Fatal("expected a refusal; the cap is 10.3GiB below the weights")
	}
	if !strings.Contains(err.Error(), MemoryLimitEnv) {
		t.Errorf("error must name the variable the operator has to change, got: %v", err)
	}
}

// ...and it must NOT be ErrLoadRequiredFull, even under requireFull. sched.go
// turns that into "evict a runner and retry", but eviction raises FreeMemory and
// the operator's cap is a constant, so every retry recomputes the same budget
// and fails identically -- evicting every other model to satisfy a constraint
// eviction cannot move.
func TestLoadDoesNotAskForEvictionOverAnOperatorCap(t *testing.T) {
	t.Setenv(MemoryLimitEnv, strconv.Itoa(8<<30))

	var c Client
	c.memory.Store(19_634_601_000)
	gpus := []ml.DeviceInfo{{FreeMemory: 88 << 30}}

	_, err := c.Load(t.Context(), ml.SystemInfo{}, gpus, true)
	if errors.Is(err, llm.ErrLoadRequiredFull) {
		t.Fatal("an operator cap must not request eviction; eviction cannot lower a constant")
	}
	if err == nil {
		t.Fatal("expected a refusal")
	}
}

// A genuine PHYSICAL shortfall must still be evictable. This is the regression
// guard on the ordering: the `available` check has to stay first, or a real
// out-of-memory stops asking the scheduler to free something and starts
// reporting the operator's knob instead.
func TestLoadStillEvictsForAPhysicalShortfall(t *testing.T) {
	var c Client
	c.memory.Store(40 << 30)
	gpus := []ml.DeviceInfo{{FreeMemory: 8 << 30}}

	_, err := c.Load(t.Context(), ml.SystemInfo{}, gpus, true)
	if !errors.Is(err, llm.ErrLoadRequiredFull) {
		t.Errorf("physical shortfall must stay evictable, got: %v", err)
	}
}
