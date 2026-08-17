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

// Unset must leave MLX's own behaviour alone. A bounded default was measured
// and withdrawn: throughput recovers only by 8 GiB, where the footprint saving
// is 497 MiB, while the whole 4.5 GB saving sits at 4 GiB and costs 15.8%
// decode. Neither trade is one to make on the operator's behalf.
func TestUnsetAppliesNoCacheLimit(t *testing.T) {
	if _, ok := parseCacheLimit(""); ok {
		t.Error("unset must not parse; configureCacheLimit then leaves MLX alone")
	}
}
