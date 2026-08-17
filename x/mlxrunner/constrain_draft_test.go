package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/x/structured"
)

// jsonDraftFixture builds a vocabulary of whole-token pieces plus an EOS, and
// a matcher for the JSON grammar. Pieces are chosen so a draft can be written
// as readable text rather than opaque ids.
func jsonDraftFixture(t *testing.T) (*structured.Vocab, [][]byte, map[string]int32, int32, *structured.Matcher) {
	t.Helper()
	texts := []string{`{`, `"`, `a`, `b`, `"`, `:`, `1`, `}`, ` `, `[`, `]`, `,`, `x`}
	pieces := make([][]byte, 0, len(texts)+1)
	byText := map[string]int32{}
	for _, s := range texts {
		if _, dup := byText[s]; dup {
			continue
		}
		byText[s] = int32(len(pieces))
		pieces = append(pieces, []byte(s))
	}
	eos := int32(len(pieces))
	pieces = append(pieces, nil) // EOS decodes to nothing

	g, err := structured.Compile([]byte(`"json"`))
	if err != nil {
		t.Fatal(err)
	}
	return structured.NewVocab(pieces, []int32{eos}), pieces, byText, eos, g.NewMatcher()
}

func ids(byText map[string]int32, ss ...string) []int32 {
	out := make([]int32, len(ss))
	for i, s := range ss {
		out[i] = byText[s]
	}
	return out
}

func isEOSFunc(eos int32) func(int32) bool {
	return func(id int32) bool { return id == eos }
}

// A draft the grammar admits end to end is accepted whole. This is the case
// that makes speculation worth doing at all.
func TestDraftPrefixAcceptsFullyLegalDraft(t *testing.T) {
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	draft := ids(byText, `{`, `"`, `a`)
	n, _ := draftPrefix(m, v, pieces, isEOSFunc(eos), draft)
	if n != len(draft) {
		t.Fatalf("accepted %d of %d legal draft tokens", n, len(draft))
	}
}

// The first illegal token truncates the draft, and everything after it is
// discarded even if it would have been legal from some other state.
//
// The token has to be chosen with care. The first version of this test used
// `{ " } a` expecting a truncation at 2, on the reasoning that `}` cannot
// close an object with an unterminated string open. It accepted all four —
// correctly: after `{"` the parser is INSIDE a string, where `}` and `a` are
// ordinary content. The grammar was right and the test was wrong, which is
// worth recording because it is the same mistake a reader will make.
//
// `:` after `{` is unambiguous: a key must come first, and no string is open
// to absorb it.
func TestDraftPrefixTruncatesAtFirstIllegalToken(t *testing.T) {
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	draft := ids(byText, `{`, `:`, `"`, `a`)
	n, _ := draftPrefix(m, v, pieces, isEOSFunc(eos), draft)
	if n != 1 {
		t.Fatalf("accepted %d, want 1 (truncate at the illegal `:`)", n)
	}
}

// Truncation mid-draft, where the prefix is long enough to be worth keeping:
// `{"a"` is a complete key, after which `,` is illegal because the value is
// still missing.
func TestDraftPrefixTruncatesMidDraft(t *testing.T) {
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	draft := ids(byText, `{`, `"`, `a`, `"`, `,`, `1`)
	n, _ := draftPrefix(m, v, pieces, isEOSFunc(eos), draft)
	if n != 4 {
		t.Fatalf("accepted %d, want 4 (keep `{\"a\"`, drop from `,`)", n)
	}
}

// A draft whose very first token is illegal yields nothing, and must not
// report a negative or partial count.
func TestDraftPrefixRejectsImmediatelyIllegalDraft(t *testing.T) {
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	draft := ids(byText, `}`, `{`)
	n, _ := draftPrefix(m, v, pieces, isEOSFunc(eos), draft)
	if n != 0 {
		t.Fatalf("accepted %d, want 0", n)
	}
}

// THE CALLER'S MATCHER MUST BE UNTOUCHED. A draft is a guess; if verification
// rejects it, the live matcher has to be exactly where it started or the
// request generates against a state its own output never reached.
func TestDraftPrefixDoesNotMutateCallersMatcher(t *testing.T) {
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	before := m.StateKey()
	if _, _ = draftPrefix(m, v, pieces, isEOSFunc(eos), ids(byText, `{`, `"`, `a`)); m.StateKey() != before {
		t.Fatal("draftPrefix advanced the caller's matcher")
	}
	// And an illegal draft must not leave it half-advanced either.
	if _, _ = draftPrefix(m, v, pieces, isEOSFunc(eos), ids(byText, `{`, `}`)); m.StateKey() != before {
		t.Fatal("a rejected draft advanced the caller's matcher")
	}
}

// The returned matcher is advanced over exactly the accepted tokens, so the
// caller can adopt it when verification commits them — that is the whole
// reason it is returned rather than discarded.
func TestDraftPrefixReturnsMatcherAdvancedOverAcceptedTokens(t *testing.T) {
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	draft := ids(byText, `{`, `"`)
	n, advanced := draftPrefix(m, v, pieces, isEOSFunc(eos), draft)
	if n != 2 {
		t.Fatalf("accepted %d, want 2", n)
	}
	// Walking the same tokens by hand must land on the same state.
	want := m.Clone()
	for _, id := range draft {
		if !want.Advance(pieces[id]) {
			t.Fatalf("hand-walk rejected token %d", id)
		}
	}
	if advanced.StateKey() != want.StateKey() {
		t.Fatal("returned matcher is not advanced over exactly the accepted tokens")
	}
}

// EOS ends the walk: it counts as accepted, and nothing after it is considered.
func TestDraftPrefixStopsAtEOS(t *testing.T) {
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	// Reach a completable state, then draft EOS with a token behind it.
	live := m.Clone()
	for _, s := range []string{`{`, `"`, `a`, `"`, `:`, `1`, `}`} {
		if !live.Advance(pieces[byText[s]]) {
			t.Fatalf("fixture could not advance over %q", s)
		}
	}
	if !live.CanComplete() {
		t.Skip("fixture grammar does not report CanComplete here")
	}
	n, _ := draftPrefix(live, v, pieces, isEOSFunc(eos), []int32{eos, byText[`{`]})
	if n != 1 {
		t.Fatalf("accepted %d, want 1 (EOS accepted, nothing after it)", n)
	}
}

// An empty draft is not an error and yields nothing to verify.
func TestDraftPrefixEmptyDraft(t *testing.T) {
	v, pieces, _, eos, m := jsonDraftFixture(t)
	n, advanced := draftPrefix(m, v, pieces, isEOSFunc(eos), nil)
	if n != 0 {
		t.Fatalf("accepted %d, want 0", n)
	}
	if advanced == nil {
		t.Fatal("must still return a usable matcher")
	}
}

// An out-of-range or undecodable id ends the walk rather than panicking. The
// serial path can treat this as impossible because the mask guarantees
// legality; here it is simply the end of what can be verified.
func TestDraftPrefixHandlesUndecodableToken(t *testing.T) {
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	n, _ := draftPrefix(m, v, pieces, isEOSFunc(eos), []int32{byText[`{`], 9999})
	if n != 1 {
		t.Fatalf("accepted %d, want 1 (stop at the out-of-range id)", n)
	}
}
