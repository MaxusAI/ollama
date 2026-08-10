package structured

import (
	"strings"
	"testing"
)

// compileRules builds a grammar from GBNF rule bodies keyed by name, rooted
// at "root", failing the test on parse errors.
func compileRules(t *testing.T, rules map[string]string) *Grammar {
	t.Helper()
	g, err := gbnfToGrammar(rules, "root")
	if err != nil {
		t.Fatalf("gbnfToGrammar: %v", err)
	}
	return g
}

func accepts(g *Grammar, s string) bool {
	m := g.NewMatcher()
	return m.Advance([]byte(s)) && m.CanComplete()
}

func TestGBNFLiteralsAndRefs(t *testing.T) {
	g := compileRules(t, map[string]string{
		"root": `"ab" mid "yz"`,
		"mid":  `"-" | "+"`,
	})
	for _, ok := range []string{"ab-yz", "ab+yz"} {
		if !accepts(g, ok) {
			t.Errorf("%q rejected", ok)
		}
	}
	for _, bad := range []string{"abyz", "ab*yz", "ab-y", "ab-yzz"} {
		if accepts(g, bad) {
			t.Errorf("%q accepted", bad)
		}
	}
}

func TestGBNFLiteralEscapes(t *testing.T) {
	g := compileRules(t, map[string]string{
		"root": `"\"" "\\" "\n" "\r"`,
	})
	if !accepts(g, "\"\\\n\r") {
		t.Error("escaped literal sequence rejected")
	}
}

func TestGBNFCharClasses(t *testing.T) {
	g := compileRules(t, map[string]string{
		"root": `[a-cx] [^0-9A]`,
	})
	for _, ok := range []string{"ab", "x!", "cz"} {
		if !accepts(g, ok) {
			t.Errorf("%q rejected", ok)
		}
	}
	for _, bad := range []string{"db", "a5", "aA"} {
		if accepts(g, bad) {
			t.Errorf("%q accepted", bad)
		}
	}
}

func TestGBNFClassEscapes(t *testing.T) {
	// \x hex escapes, escaped backslash, escaped ']', escaped '-'.
	g := compileRules(t, map[string]string{
		"root": `[\x41-\x43] [\\\]] [\-a]`,
	})
	for _, ok := range []string{`B\-`, `C]a`} {
		if !accepts(g, ok) {
			t.Errorf("%q rejected", ok)
		}
	}
	if accepts(g, `D\-`) {
		t.Error("out-of-range hex class byte accepted")
	}
}

func TestGBNFRepetition(t *testing.T) {
	g := compileRules(t, map[string]string{
		"root": `"a"? "b"* "c"+ [d]{2} [e]{1,3} "f"{2,}`,
	})
	for _, ok := range []string{"cddeff", "abbbccddeeefff", "acddeff"} {
		if !accepts(g, ok) {
			t.Errorf("%q rejected", ok)
		}
	}
	for _, bad := range []string{"ddeff", "cdeff", "cddff", "cddeeeef", "cddef", "aacddeff"} {
		if accepts(g, bad) {
			t.Errorf("%q accepted", bad)
		}
	}
}

func TestGBNFRepetitionAppliesToWholeLiteral(t *testing.T) {
	// llama.cpp's GBNF parser applies postfix operators to the entire
	// preceding literal, not just its last character.
	g := compileRules(t, map[string]string{
		"root": `"ab"*`,
	})
	for _, ok := range []string{"", "ab", "abab"} {
		if !accepts(g, ok) {
			t.Errorf("%q rejected", ok)
		}
	}
	for _, bad := range []string{"a", "abb", "aab"} {
		if accepts(g, bad) {
			t.Errorf("%q accepted", bad)
		}
	}
}

func TestGBNFGroupsAndAlternation(t *testing.T) {
	g := compileRules(t, map[string]string{
		"root": `("x" | "y" "z"){2} ( "!" )?`,
	})
	for _, ok := range []string{"xx", "xyz", "yzyz!", "yzx"} {
		if !accepts(g, ok) {
			t.Errorf("%q rejected", ok)
		}
	}
	for _, bad := range []string{"x", "yz", "xyzx", "!"} {
		if accepts(g, bad) {
			t.Errorf("%q accepted", bad)
		}
	}
}

func TestGBNFRepetitionThreshold(t *testing.T) {
	// Repetitions expand to one copy of the body per repetition, so an
	// unbounded count is a memory-exhaustion lever. Parity with llama.cpp:
	// a single count over MAX_REPETITION_THRESHOLD is rejected, and so is a
	// count whose product with the rules the unit already generated reaches
	// it — the guard that stops nested repetitions from multiplying out.
	cases := []struct {
		name string
		body string
		ok   bool
	}{
		{"just under the threshold", `[a]{0,1999}`, true},
		{"at the threshold", `[a]{0,2000}`, false},
		{"max far over", `[a]{0,300000000}`, false},
		{"min far over", `[a]{300000000,}`, false},
		{"exact count over", `[a]{300000000}`, false},
		{"alternation is not a repetition", `("x" | "y"){0,100}`, true},
		{"nested repetitions multiply", `("x"{0,1000}){0,10}`, false},
	}
	for _, c := range cases {
		_, err := gbnfToGrammar(map[string]string{"root": c.body}, "root")
		if c.ok && err != nil {
			t.Errorf("%s: %s: unexpected error: %v", c.name, c.body, err)
		}
		if !c.ok {
			if err == nil {
				t.Errorf("%s: %s: expected an error", c.name, c.body)
			} else if !strings.Contains(err.Error(), "repetition") {
				t.Errorf("%s: %s: error %q does not mention repetitions", c.name, c.body, err)
			}
		}
	}
}

func TestGBNFCanonicalizeMergesEqualRules(t *testing.T) {
	// Rules that differ only in name compile to distinct rules, and keeping
	// both alive forks the stack set for no reason; equal rules are merged
	// and the duplicate alternatives they leave behind are dropped.
	g := compileRules(t, map[string]string{
		"root": `rep | rep2`,
		"rep":  `"a"* "b"`,
		"rep2": `"a"* "b"`,
	})
	if got := len(g.rules[g.root].alts); got != 1 {
		t.Errorf("root has %d alternatives, want 1 after merging equal rules", got)
	}
	for _, ok := range []string{"b", "ab", "aaab"} {
		if !accepts(g, ok) {
			t.Errorf("%q rejected", ok)
		}
	}
	for _, bad := range []string{"", "a", "ba"} {
		if accepts(g, bad) {
			t.Errorf("%q accepted", bad)
		}
	}
}

func TestGBNFUndefinedRuleErrors(t *testing.T) {
	if _, err := gbnfToGrammar(map[string]string{"root": `missing`}, "root"); err == nil {
		t.Error("expected error for undefined rule reference")
	}
	if _, err := gbnfToGrammar(map[string]string{"other": `"x"`}, "root"); err == nil {
		t.Error("expected error for missing root")
	}
}

func TestGBNFRecursion(t *testing.T) {
	// Balanced parens via recursion exercises ref push/pop.
	g := compileRules(t, map[string]string{
		"root": `"(" root ")" | "x"`,
	})
	for _, ok := range []string{"x", "(x)", "(((x)))"} {
		if !accepts(g, ok) {
			t.Errorf("%q rejected", ok)
		}
	}
	for _, bad := range []string{"()", "((x)", "(x))"} {
		if accepts(g, bad) {
			t.Errorf("%q accepted", bad)
		}
	}
}
