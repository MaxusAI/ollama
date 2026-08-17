package mlxrunner

import "testing"

// Zero is the most useful cache limit there is -- "retain nothing" -- and it
// must not be confused with an absent value. parseMemoryBudget rejects 0
// because a zero MEMORY budget forbids every allocation; reusing it here made
// OLLAMA_MLX_CACHE_LIMIT=0 a no-op that silently measured the default.
func TestParseCacheLimitAdmitsZero(t *testing.T) {
	got, ok := parseCacheLimit("0")
	if !ok {
		t.Fatal("0 must be accepted: it means retain nothing")
	}
	if got != 0 {
		t.Errorf("got %d, want 0", got)
	}
}

func TestParseCacheLimitRejectsAbsentAndJunk(t *testing.T) {
	for _, s := range []string{"", "abc", "-1", "18446744073709551615999"} {
		if _, ok := parseCacheLimit(s); ok {
			t.Errorf("%q must not be accepted as a cache limit", s)
		}
	}
}

// A memory budget of 0 is still rejected: the two parsers differ ONLY on zero.
func TestZeroIsStillInvalidAsAMemoryBudget(t *testing.T) {
	if _, ok := parseMemoryBudget("0"); ok {
		t.Error("a zero memory budget would forbid every allocation")
	}
}

// Unset must apply the bounded default, not MLX's own -- which is derived from
// TOTAL device memory (90.22 GiB on the measured card) and is the behaviour
// this change exists to replace.
func TestDefaultCacheLimitIsBounded(t *testing.T) {
	if DefaultCacheLimit <= 0 {
		t.Fatal("the default must bound the cache")
	}
	if DefaultCacheLimit > 16<<30 {
		t.Errorf("DefaultCacheLimit %d is not a bound worth having", DefaultCacheLimit)
	}
	// The env parser must not claim an unset value, so configureCacheLimit
	// falls through to the default rather than skipping.
	if _, ok := parseCacheLimit(""); ok {
		t.Error("empty must not parse; the default applies instead")
	}
}
