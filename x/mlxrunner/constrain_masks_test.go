package mlxrunner

import (
	"testing"
)

// Row i predicts draft i, so row i must carry the mask of the state BEFORE
// that draft. This is the property the whole scheme rests on: get the
// off-by-one wrong and verification compares against a mask from the wrong
// position, which would reject legal drafts or — worse — admit illegal ones.
func TestDraftMasksRowIPrecedesDraftI(t *testing.T) {
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	draft := ids(byText, `{`, `"`, `a`)
	masks, legal, _ := draftMasks(m, v, pieces, isEOSFunc(eos), draft)
	if legal != 3 {
		t.Fatalf("legal=%d, want 3", legal)
	}
	if len(masks) != legal+1 {
		t.Fatalf("got %d masks, want legal+1=%d", len(masks), legal+1)
	}
	// Row 0 is the pre-draft state: `{` legal, `a` not (an object cannot open
	// with a bare letter).
	if !masks[0].Allowed(byText[`{`]) {
		t.Error("row 0 must admit `{` from the initial state")
	}
	if masks[0].Allowed(byText[`a`]) {
		t.Error("row 0 must not admit a bare `a` from the initial state")
	}
	// Hand-walk to the state before draft 2 and compare admissibility of the
	// token that draft actually is.
	want := m.Clone()
	want.Advance(pieces[byText[`{`]])
	want.Advance(pieces[byText[`"`]])
	if masks[2].Allowed(byText[`a`]) != v.Mask(want).Allowed(byText[`a`]) {
		t.Error("row 2 mask does not match the state after drafts[0..1]")
	}
}

// A rejected draft still yields the masks for the rows up to and including the
// rejection, because verification needs the mask AT the rejected position to
// sample a replacement there.
func TestDraftMasksIncludeTheRejectingRow(t *testing.T) {
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	draft := ids(byText, `{`, `:`, `"`)
	masks, legal, _ := draftMasks(m, v, pieces, isEOSFunc(eos), draft)
	if legal != 1 {
		t.Fatalf("legal=%d, want 1", legal)
	}
	// Rows: 0 (before `{`) and 1 (before the illegal `:`) — the rejecting row
	// is present, and there is no bonus row beyond it.
	if len(masks) != 2 {
		t.Fatalf("got %d masks, want 2", len(masks))
	}
	if masks[1].Allowed(byText[`:`]) {
		t.Error("the rejecting row must not admit the token that was rejected")
	}
}

// EOS admits no bonus row: nothing follows it, so there is no state at which a
// further token could be sampled.
func TestDraftMasksNoBonusRowAfterEOS(t *testing.T) {
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	live := m.Clone()
	for _, s := range []string{`{`, `"`, `a`, `"`, `:`, `1`, `}`} {
		if !live.Advance(pieces[byText[s]]) {
			t.Fatalf("fixture could not advance over %q", s)
		}
	}
	if !live.CanComplete() {
		t.Skip("fixture grammar does not report CanComplete here")
	}
	masks, legal, _ := draftMasks(live, v, pieces, isEOSFunc(eos), []int32{eos})
	if legal != 1 {
		t.Fatalf("legal=%d, want 1", legal)
	}
	if len(masks) != 1 {
		t.Fatalf("got %d masks, want 1 (no bonus row past EOS)", len(masks))
	}
}

// The caller's matcher is untouched, and the returned one is advanced over
// exactly the admitted drafts — same contract as draftPrefix, asserted
// separately because this walk is a second implementation of it.
func TestDraftMasksMatcherContract(t *testing.T) {
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	before := m.StateKey()
	draft := ids(byText, `{`, `"`)
	_, legal, advanced := draftMasks(m, v, pieces, isEOSFunc(eos), draft)
	if m.StateKey() != before {
		t.Fatal("draftMasks advanced the caller's matcher")
	}
	want := m.Clone()
	for _, id := range draft[:legal] {
		want.Advance(pieces[id])
	}
	if advanced.StateKey() != want.StateKey() {
		t.Fatal("returned matcher is not advanced over exactly the admitted drafts")
	}
}

// draftMasks and draftPrefix must agree on how much of a draft is admissible.
// They walk the grammar independently, so a divergence means one of them is
// wrong and verification would disagree with the truncation that fed it.
func TestDraftMasksAgreesWithDraftPrefix(t *testing.T) {
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	for _, draft := range [][]int32{
		ids(byText, `{`, `"`, `a`),
		ids(byText, `{`, `:`, `"`),
		ids(byText, `}`),
		ids(byText, `{`, `"`, `a`, `"`, `,`),
		{},
	} {
		nPrefix, mPrefix := draftPrefix(m, v, pieces, isEOSFunc(eos), draft)
		_, nMasks, mMasks := draftMasks(m, v, pieces, isEOSFunc(eos), draft)
		if nPrefix != nMasks {
			t.Errorf("draft %v: draftPrefix=%d draftMasks=%d", draft, nPrefix, nMasks)
		}
		if mPrefix.StateKey() != mMasks.StateKey() {
			t.Errorf("draft %v: the two walks ended in different states", draft)
		}
	}
}

// An empty draft still yields the one row the bonus token is sampled from.
func TestDraftMasksEmptyDraftHasBonusRow(t *testing.T) {
	v, pieces, _, eos, m := jsonDraftFixture(t)
	masks, legal, _ := draftMasks(m, v, pieces, isEOSFunc(eos), nil)
	if legal != 0 {
		t.Fatalf("legal=%d, want 0", legal)
	}
	if len(masks) != 1 {
		t.Fatalf("got %d masks, want 1 (the bonus row)", len(masks))
	}
}
