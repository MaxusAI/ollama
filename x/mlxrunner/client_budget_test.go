package mlxrunner

import (
	"strconv"
	"testing"
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
