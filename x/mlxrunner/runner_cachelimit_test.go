package mlxrunner

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
)

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

// A malformed value must SAY so. Unset is silent by design; unparseable is not,
// because a discarded request that leaves no trace is the exact defect the
// override work exists to remove -- a sweep over several values produces
// identical runs and nothing in the output explains why. That is how the cap-0
// bug hid, reporting the default's footprint under the cap-0 label.
//
// This path returns before touching MLX, so unlike the C-call tests it runs
// everywhere, including CI without a GPU.
func TestMalformedCacheLimitIsReported(t *testing.T) {
	var buf bytes.Buffer
	original := slog.Default()
	t.Cleanup(func() { slog.SetDefault(original) })
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))

	// "8GiB" is the plausible thing to type: the comment above CacheLimitEnv
	// tells an operator to set it, and nothing there says decimal bytes only.
	t.Setenv(CacheLimitEnv, "8GiB")
	configureCacheLimit()

	out := buf.String()
	if !strings.Contains(out, CacheLimitEnv) {
		t.Errorf("a malformed value must be reported and must name the variable; got: %q", out)
	}
	if !strings.Contains(out, "8GiB") {
		t.Errorf("the rejected value must appear so the operator can see what was read; got: %q", out)
	}
}

// ...and unset must stay silent, or every default run gains a warning.
func TestUnsetCacheLimitStaysSilent(t *testing.T) {
	var buf bytes.Buffer
	original := slog.Default()
	t.Cleanup(func() { slog.SetDefault(original) })
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))

	t.Setenv(CacheLimitEnv, "")
	configureCacheLimit()

	if buf.Len() != 0 {
		t.Errorf("unset must be silent, got: %q", buf.String())
	}
}
