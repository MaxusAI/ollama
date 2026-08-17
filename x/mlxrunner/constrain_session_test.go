package mlxrunner

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/x/structured"
)

// session builds a speculationSession carrying only the grammar state, which
// is all grammarRows and adoptGrammar touch. No Runner, no model, no MLX --
// isEOS is a field for exactly this reason, so the bookkeeping that decides
// whether the format holds is checkable on any host.
func session(t *testing.T) (*speculationSession, *structured.Vocab, [][]byte, map[string]int32, int32) {
	t.Helper()
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	return &speculationSession{matcher: m, vocab: v, pieces: pieces, isEOS: isEOSFunc(eos)}, v, pieces, byText, eos
}

// sameMask compares two masks by what they admit rather than by identity, so
// the assertion does not depend on Vocab's memoization returning one instance.
func sameMask(a, b *structured.Mask, size int32) bool {
	for id := int32(0); id < size; id++ {
		if a.Allowed(id) != b.Allowed(id) {
			return false
		}
	}
	return true
}

// Without a matcher every masked branch must be inert: nil masks, and the full
// draft reported as admitted. This is what keeps the unconstrained path
// byte-identical to what it was before grammar-aware speculation existed.
func TestGrammarRowsInertWithoutMatcher(t *testing.T) {
	s := &speculationSession{}
	masks, admitted := s.grammarRows([]int32{1, 2, 3})
	if masks != nil {
		t.Errorf("got %d masks, want nil without a matcher", len(masks))
	}
	if admitted != 3 {
		t.Errorf("admitted=%d, want 3 (the whole draft)", admitted)
	}
}

// THE BONUS-ROW CONTRACT accept() keys off. A normal round has one mask per
// draft plus the row the bonus token is sampled from; a draft ending in an
// admitted EOS has no such row, because nothing follows an EOS. accept()
// detects that case as len(masks) == admitted and drops the EOS rather than
// running a round whose last row does not exist.
func TestGrammarRowsBonusRowPresentExceptAfterEOS(t *testing.T) {
	s, _, pieces, byText, eos := session(t)

	masks, admitted := s.grammarRows(ids(byText, `{`, `"`, `a`))
	if admitted != 3 || len(masks) != admitted+1 {
		t.Fatalf("legal draft: admitted=%d len(masks)=%d, want 3 and 4", admitted, len(masks))
	}

	// Reach a completable state, then draft EOS: the mask sequence ends with
	// the row BEFORE the EOS and there is no bonus row.
	for _, s2 := range []string{`{`, `"`, `a`, `"`, `:`, `1`, `}`} {
		if !s.matcher.Advance(pieces[byText[s2]]) {
			t.Fatalf("fixture could not advance over %q", s2)
		}
	}
	if !s.matcher.CanComplete() {
		t.Skip("fixture grammar does not report CanComplete here")
	}
	masks, admitted = s.grammarRows([]int32{eos})
	if admitted != 1 || len(masks) != admitted {
		t.Fatalf("EOS draft: admitted=%d len(masks)=%d, want 1 and 1 (no bonus row)", admitted, len(masks))
	}
}

// adoptGrammar leaves the matcher where a hand-walk over the same tokens does.
func TestAdoptGrammarAdvancesOverEmittedTokens(t *testing.T) {
	s, _, pieces, byText, _ := session(t)
	want := s.matcher.Clone()
	emitted := ids(byText, `{`, `"`, `a`)
	for _, id := range emitted {
		want.Advance(pieces[id])
	}
	if err := s.adoptGrammar(emitted); err != nil {
		t.Fatal(err)
	}
	if s.matcher.StateKey() != want.StateKey() {
		t.Fatal("matcher is not advanced over exactly the emitted tokens")
	}
}

// EOS terminates the walk: it has no piece to advance over, and nothing after
// it is emitted.
func TestAdoptGrammarStopsAtEOS(t *testing.T) {
	s, _, _, byText, eos := session(t)
	before := s.matcher.StateKey()
	if err := s.adoptGrammar([]int32{eos, byText[`{`]}); err != nil {
		t.Fatal(err)
	}
	if s.matcher.StateKey() != before {
		t.Fatal("adoptGrammar advanced past an EOS")
	}
}

// AN ILLEGAL TOKEN MUST BE AN ERROR. Every token reaching adoptGrammar was
// drawn from a masked distribution, so a failure to advance means the matcher
// has drifted from the emitted text -- and continuing from there ships
// malformed output under a format promise. Silently discarding the failure is
// what let the drift this guard catches go unnoticed.
func TestAdoptGrammarRejectsIllegalToken(t *testing.T) {
	// `{` then `:` -- a key must come first, and no string is open to absorb
	// the colon. (`{` then `}` would NOT do: an empty object is valid JSON.
	// The same trap is recorded in constrain_draft_test.go.)
	s, _, _, byText, _ := session(t)
	err := s.adoptGrammar(ids(byText, `{`, `:`))
	if err == nil {
		t.Fatal("adoptGrammar accepted a token the grammar forbids")
	}
	if !strings.Contains(err.Error(), "illegal token") {
		t.Errorf("error does not name the failure: %v", err)
	}

	// An id with no decodable piece is the same failure, not a silent skip.
	s2, _, _, _, _ := session(t)
	if err := s2.adoptGrammar([]int32{9999}); err == nil {
		t.Fatal("adoptGrammar accepted an out-of-range id")
	}
}

// Without a matcher there is nothing to advance and nothing to fail.
func TestAdoptGrammarWithoutMatcherIsNoOp(t *testing.T) {
	s := &speculationSession{}
	if err := s.adoptGrammar([]int32{1, 2, 3}); err != nil {
		t.Fatalf("adoptGrammar without a matcher returned %v", err)
	}
}

// THE INVARIANT THE WHOLE FIX RESTS ON: after every round, the matcher
// describes every token emitted so far, so mask row 0 of the next round is the
// state after the token held as current.
//
// This is the property the gated path violated in three places at once --
// parked tokens, resumed tokens, and the residual/bonus token were all emitted
// without being adopted. The failure is not a one-off skew: row 0 is then the
// state after some earlier prefix, drafts are judged against it, legal ones
// look illegal, the round parks, and the parked token is emitted unadopted
// too. It compounds rather than converging, which is why it presented as
// output with no constraint at all rather than as slightly-wrong output.
func TestAdoptedMatcherTracksEveryEmittedToken(t *testing.T) {
	s, v, pieces, byText, _ := session(t)
	hand := s.matcher.Clone()

	// Rounds of varying shape: a multi-token accept, a single token (as a
	// parked or resumed step emits), then another run.
	rounds := [][]int32{
		ids(byText, `{`, `"`),
		ids(byText, `a`),
		ids(byText, `"`, `:`),
		ids(byText, `1`),
	}
	for i, emitted := range rounds {
		if err := s.adoptGrammar(emitted); err != nil {
			t.Fatalf("round %d: %v", i, err)
		}
		for _, id := range emitted {
			if !hand.Advance(pieces[id]) {
				t.Fatalf("round %d: hand-walk rejected token %d", i, id)
			}
		}
		masks, admitted := s.grammarRows(nil)
		if admitted != 0 || len(masks) != 1 {
			t.Fatalf("round %d: admitted=%d len(masks)=%d, want 0 and 1", i, admitted, len(masks))
		}
		if !sameMask(masks[0], v.Mask(hand), int32(len(pieces))) {
			t.Fatalf("round %d: row 0 is not the state after every emitted token", i)
		}
	}
}
