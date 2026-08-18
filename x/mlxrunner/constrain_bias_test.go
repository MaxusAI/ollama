package mlxrunner

import (
	"math"
	"os"
	"testing"
)

// The bias must be row-major and contiguous, one row per verification row,
// because mlx.FromValues takes a flat slice and applies the shape on top. A
// transposed or short layout would be read as valid data and silently mask the
// wrong positions.
func TestFillMaskBiasLayout(t *testing.T) {
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	masks, legal, _ := draftMasks(m, v, pieces, isEOSFunc(eos), ids(byText, `{`, `"`))
	if legal != 2 {
		t.Fatalf("legal=%d, want 2", legal)
	}
	vocabDim := len(pieces)
	buf := fillMaskBias(masks, vocabDim, nil)

	if len(buf) != len(masks)*vocabDim {
		t.Fatalf("len=%d, want rows*vocabDim=%d", len(buf), len(masks)*vocabDim)
	}
	for row, mask := range masks {
		for id := range vocabDim {
			got := buf[row*vocabDim+id]
			if mask.Allowed(int32(id)) {
				if got != 0 {
					t.Fatalf("row %d id %d: allowed but bias %v, want 0", row, id, got)
				}
			} else if !math.IsInf(float64(got), -1) {
				t.Fatalf("row %d id %d: disallowed but bias %v, want -Inf", row, id, got)
			}
		}
	}
}

// Rows must be independent. The first version of the allowed-set walk could
// have leaked a previous row's zeros forward by reusing an offset; this pins
// that each row is masked by its own state and nothing else.
func TestFillMaskBiasRowsAreIndependent(t *testing.T) {
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	masks, _, _ := draftMasks(m, v, pieces, isEOSFunc(eos), ids(byText, `{`, `"`))
	vocabDim := len(pieces)
	buf := fillMaskBias(masks, vocabDim, nil)

	// `{` is admissible at row 0 (initial state) and not at row 1 (inside an
	// object, before a key) — so the two rows must differ at that id.
	open := int(byText[`{`])
	r0, r1 := buf[0*vocabDim+open], buf[1*vocabDim+open]
	if r0 == r1 && masks[0].Allowed(byText[`{`]) != masks[1].Allowed(byText[`{`]) {
		t.Fatal("rows carry the same bias where their masks differ")
	}
}

// The buffer is reused across rounds, so a shorter follow-up must not leave a
// previous round's data visible past its own length, and a longer one must
// grow rather than truncate.
func TestFillMaskBiasReusesBufferSafely(t *testing.T) {
	v, pieces, byText, eos, m := jsonDraftFixture(t)
	vocabDim := len(pieces)

	long, _, _ := draftMasks(m, v, pieces, isEOSFunc(eos), ids(byText, `{`, `"`, `a`))
	buf := fillMaskBias(long, vocabDim, nil)
	grown := cap(buf)

	short, _, _ := draftMasks(m, v, pieces, isEOSFunc(eos), ids(byText, `{`))
	buf = fillMaskBias(short, vocabDim, buf)
	if len(buf) != len(short)*vocabDim {
		t.Fatalf("reused buffer len=%d, want %d", len(buf), len(short)*vocabDim)
	}
	if cap(buf) < grown {
		t.Error("buffer shrank; the point of passing it back is to keep the allocation")
	}
	// And the reused prefix must be correct, not left over.
	for row, mask := range short {
		for id := range vocabDim {
			got := buf[row*vocabDim+id]
			if mask.Allowed(int32(id)) != (got == 0) {
				t.Fatalf("row %d id %d: stale bias %v after reuse", row, id, got)
			}
		}
	}
}

// THE GATE IS OFF UNLESS EXPLICITLY SET. This mechanism has never run against a
// real model and it edits the path where a mistake is silent KV corruption, so
// anything other than an explicit opt-in must leave the serial path in place.
func TestGrammarSpeculationDefaultsOff(t *testing.T) {
	t.Setenv(GrammarSpeculationEnv, "")
	if grammarSpeculationEnabled() {
		t.Error("unset must be off")
	}
	for _, v := range []string{"0", "no", "off", "yes", "2", "TRUE1"} {
		t.Setenv(GrammarSpeculationEnv, v)
		if grammarSpeculationEnabled() {
			t.Errorf("%q must not enable the gate", v)
		}
	}
	for _, v := range []string{"1", "true", "TRUE", "True"} {
		t.Setenv(GrammarSpeculationEnv, v)
		if !grammarSpeculationEnabled() {
			t.Errorf("%q must enable the gate", v)
		}
	}
	os.Unsetenv(GrammarSpeculationEnv)
}
