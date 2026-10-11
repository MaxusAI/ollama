package mlxrunner

import (
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// draft_num_predict on MLX (the fork): the request's value reaches the runner as
// CompletionRequest.DraftLimit. nil keeps the adaptive depth the runner always
// chose; 0 parks the drafter for that request; N caps the depth at N. None of
// these reload the model -- the drafter stays loaded either way -- so one loaded
// drafting tag measures drafted and undrafted generation back to back.

func TestRequestDraftLimit(t *testing.T) {
	for _, tc := range []struct {
		name string
		opt  int
		want *int
	}{
		// The server sends -1 for an MLX model whose request and Modelfile
		// leave draft_num_predict unset (modelOptionsWithEmbeddingBatchDefault).
		{"unset keeps the adaptive depth", -1, nil},
		{"0 turns drafting off", 0, new(0)},
		{"N caps the depth", 6, new(6)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := requestDraftLimit(api.Options{Runner: api.Runner{DraftNumPredict: tc.opt}})
			if (got == nil) != (tc.want == nil) || (got != nil && *got != *tc.want) {
				t.Fatalf("requestDraftLimit(%d) = %v, want %v", tc.opt, deref(got), deref(tc.want))
			}
		})
	}
}

func deref(p *int) any {
	if p == nil {
		return nil
	}
	return *p
}

func TestDraftLimitGatesTheRequest(t *testing.T) {
	for _, tc := range []struct {
		name  string
		limit *int
		want  bool
	}{
		{"no limit drafts", nil, true},
		{"limit 0 parks", new(0), false},
		{"a positive limit drafts", new(3), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := Request{}
			r.DraftLimit = tc.limit
			if got := draftingEnabled(r); got != tc.want {
				t.Fatalf("draftingEnabled() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestDraftLimitCapsTheDepth(t *testing.T) {
	s := &speculation{drafter: grammarTestDrafter{}, depth: newDepthController()}
	s.depth.scheduled = 9
	// Every depth costs the same, so with full acceptance deeper always pays
	// and the controller climbs as fast as its acceptance frontier allows.
	for n := 0; n <= 16; n++ {
		s.depth.cost.observe(n, time.Millisecond)
	}

	adaptive := s.open(Request{}, nil)
	defer adaptive.close()
	if adaptive.limit != 9 {
		t.Fatalf("unlimited request opened at depth %d, want the scheduled 9", adaptive.limit)
	}

	r := Request{}
	r.DraftLimit = new(2)
	capped := s.open(r, nil)
	defer capped.close()
	if capped.limit != 2 {
		t.Fatalf("limited request opened at depth %d, want its limit 2", capped.limit)
	}

	// Every round accepts in full, so the controller keeps reaching deeper; the
	// capped session must never follow it past its limit, and the uncapped one
	// must (or this test could not fail).
	deeper := false
	for range 200 {
		capped.endRound(capped.limit, capped.limit, capped.limit)
		if capped.limit > 2 {
			t.Fatalf("limited request's depth rose to %d, past its limit 2", capped.limit)
		}
		adaptive.endRound(adaptive.limit, adaptive.limit, adaptive.limit)
		deeper = deeper || adaptive.limit > 2
	}
	if !deeper {
		t.Fatal("the unlimited session never went past depth 2; the cap is untested")
	}
}
