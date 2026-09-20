package xgrammar_test

// Characterisation of upstream's xgrammar engine on the points where the
// fork's retired pure-Go sampler (x/structured, deleted 2026-09-17) behaved
// differently. Each test passes while the measured behaviour holds and fails
// when an xgrammar bump changes it — that failure is the signal to update ADR
// 0033 and the retirement register, not a defect. The measurements come from
// the parity gate that ran both engines side by side before the deletion
// (xgrammar v0.2.5, byte-level vocabulary, 108 agreements, 0 regressions).
//
// A byte-level vocabulary — one piece per byte value plus EOS — lets a text be
// accepted one byte per token, so the verdicts here are about grammars, not
// tokenisation. Runs wherever libollama_xgrammar.so can be found: beside the
// MLX payload, or in OLLAMA_XGRAMMAR_LIBDIR.

import (
	"errors"
	"io/fs"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/mlxrunner/xgrammar"
)

const (
	byteVocabSize       = 257
	byteEOS       int32 = 256
)

func byteVocabulary() []string {
	pieces := make([]string, byteVocabSize)
	for i := range 256 {
		pieces[i] = string([]byte{byte(i)})
	}
	pieces[byteEOS] = "<eos>"
	return pieces
}

func byteCompiler(t testing.TB) *xgrammar.Compiler {
	t.Helper()
	dir := os.Getenv("OLLAMA_XGRAMMAR_LIBDIR")
	if dir == "" {
		dir = testLibraryDir(t)
	}
	compiler, err := xgrammar.New(dir, byteVocabulary(), byteVocabSize, []int32{byteEOS}, 8, 128<<20)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			t.Skipf("native xgrammar payload is not built: %v", err)
		}
		t.Fatal(err)
	}
	t.Cleanup(compiler.Close)
	return compiler
}

// verdict is what the engine says about one text: the index of the first
// rejected byte (-1 when every byte was accepted) and whether it could end there.
type verdict struct {
	reject   int
	complete bool
}

func (v verdict) ok() bool { return v.reject < 0 && v.complete }

func (v verdict) String() string {
	if v.reject >= 0 {
		return "reject@" + strconv.Itoa(v.reject)
	}
	if v.complete {
		return "accept,complete"
	}
	return "accept,incomplete"
}

// accepts drives in through a fresh matcher for schema one byte per token,
// gated by the mask exactly as a decoder would be.
func accepts(t *testing.T, compiler *xgrammar.Compiler, schema, in string) verdict {
	t.Helper()
	matcher, err := compiler.Compile(schemaTag(schema))
	if err != nil {
		t.Fatalf("compile %s: %v", schema, err)
	}
	defer matcher.Close()
	row := make([]int32, (byteVocabSize+31)/32)
	next := func() bool {
		constrained, err := matcher.Fill(row)
		if err != nil {
			t.Fatal(err)
		}
		return constrained
	}
	for i := range len(in) {
		id := int32(in[i])
		if next() && !allowed(row, id) {
			return verdict{reject: i}
		}
		if err := matcher.Accept(id); err != nil {
			return verdict{reject: i}
		}
	}
	if matcher.Terminated() {
		return verdict{reject: -1, complete: true}
	}
	constrained := next()
	return verdict{reject: -1, complete: !constrained || allowed(row, byteEOS)}
}

// changed is the message every pin fails with: the behaviour moved, so the
// record must move with it.
func changed(t *testing.T, what, schema, in string, got verdict) {
	t.Helper()
	t.Errorf("xgrammar's behaviour changed: %s\n  schema %s\n  input  %q -> %s\n"+
		"  Update ADR 0033 and docs/maxusai/retirement-register.md, then retire or rewrite this pin.", what, schema, in, got)
}

// TestKnownBehaviourAllOfWithSeveralBranchesIsPermissive pins the one place
// the retired sampler constrained more than upstream's engine: on xgrammar
// v0.2.5 an allOf with more than one branch enforces neither the branches'
// required properties nor their types nor property order, and the converter
// warns that allOf support is still ongoing. A caller wanting those
// constraints on the MLX path must write one properties block.
func TestKnownBehaviourAllOfWithSeveralBranchesIsPermissive(t *testing.T) {
	compiler := byteCompiler(t)
	open := `{"type":"object","allOf":[{"properties":{"a":{"type":"integer"}},"required":["a"]},{"properties":{"b":{"type":"string"}},"required":["b"]}]}`
	closed := `{"type":"object","allOf":[{"properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false},{"properties":{"b":{"type":"string"}},"required":["b"],"additionalProperties":false}]}`
	for _, c := range []struct{ schema, in, why string }{
		{open, `{"a":1,"b":"s"}`, "the valid instance is accepted"},
		{open, `{"a":1}`, "a required property from the second branch is not required"},
		{open, `{"b":"s"}`, "a required property from the first branch is not required"},
		{open, `{"a":"1","b":"s"}`, "a branch's property type is not enforced"},
		{closed, `{}`, "closing every branch changes nothing"},
	} {
		if got := accepts(t, compiler, c.schema, c.in); !got.ok() {
			changed(t, "allOf is now enforced ("+c.why+")", c.schema, c.in, got)
		}
	}
	// The same constraints in one properties block are enforced, which is the
	// workaround the pin exists to point at.
	merged := `{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string"}},"required":["a","b"],"additionalProperties":false}`
	for _, in := range []string{`{"a":1}`, `{"a":"1","b":"s"}`, `{}`} {
		if got := accepts(t, compiler, merged, in); got.ok() {
			changed(t, "a single properties block no longer enforces required and types", merged, in, got)
		}
	}
}

// TestKnownBehaviourIntegerGrammarRefusesNegativeZero: xgrammar's integer
// grammar has no -0; its number grammar accepts it. Semantically zero either
// way, so nothing depends on it, but a bump that admits -0 changes what a
// bounded-integer field can emit.
func TestKnownBehaviourIntegerGrammarRefusesNegativeZero(t *testing.T) {
	compiler := byteCompiler(t)
	integer := `{"type":"object","properties":{"v":{"type":"integer"}},"required":["v"]}`
	number := `{"type":"object","properties":{"v":{"type":"number"}},"required":["v"]}`
	if got := accepts(t, compiler, integer, `{"v":-0}`); got.ok() {
		changed(t, "the integer grammar now accepts -0", integer, `{"v":-0}`, got)
	}
	if got := accepts(t, compiler, number, `{"v":-0}`); !got.ok() {
		changed(t, "the number grammar now refuses -0", number, `{"v":-0}`, got)
	}
	if got := accepts(t, compiler, integer, `{"v":0}`); !got.ok() {
		changed(t, "the integer grammar refuses 0", integer, `{"v":0}`, got)
	}
}

// TestKnownBehaviourWhitespaceBeforeColonIsAccepted: llama.cpp's grammar, which
// the retired sampler ported, never allowed whitespace between a key and its
// colon; xgrammar's JSON grammar does. Unbounded whitespace elsewhere is ADR
// 0035's subject, not this pin's.
func TestKnownBehaviourWhitespaceBeforeColonIsAccepted(t *testing.T) {
	compiler := byteCompiler(t)
	schema := `{"type":"object"}` // what requestGrammar sends for format "json"
	if got := accepts(t, compiler, schema, "{\"a\" :1}"); !got.ok() {
		changed(t, "whitespace before a colon is now refused", schema, "{\"a\" :1}", got)
	}
}

// TestUnboundedRepetitionCompilesWithinBudget is ADR 0013's property on the
// upstream engine, kept after the guard that enforced it was deleted with
// x/structured: a schema whose repetition bounds would expand to unbounded
// work must not cost unbounded time or memory to compile. xgrammar compiles
// repetition lazily; measured 3–30 ms within 1 MiB for every schema below.
func TestUnboundedRepetitionCompilesWithinBudget(t *testing.T) {
	compiler := byteCompiler(t)
	for _, schema := range []string{
		`{"type":"string","maxLength":300000000}`,
		`{"type":"string","minLength":300000000}`,
		`{"type":"array","items":{"type":"integer"},"maxItems":300000000}`,
		`{"type":"array","items":{"type":"integer"},"minItems":300000000}`,
		`{"type":"array","maxItems":1999,"items":{"type":"string","maxLength":1999}}`,
	} {
		before := rssBytes()
		start := time.Now()
		m, err := compiler.Compile(schemaTag(schema))
		took := time.Since(start)
		if m != nil {
			m.Close()
		}
		grew := rssBytes() - before
		outcome := "compiled"
		if err != nil {
			outcome = "refused: " + err.Error()
		}
		t.Logf("%s: %s in %s, RSS %+d MiB", schema, outcome, took.Round(time.Millisecond), grew>>20)
		if took > 30*time.Second {
			t.Errorf("%s: took %s; the bound ADR 0013 protected is broken on this engine", schema, took)
		}
		if grew > 1<<30 {
			t.Errorf("%s: RSS grew by %d MiB; the bound ADR 0013 protected is broken on this engine", schema, grew>>20)
		}
	}
}

func rssBytes() int64 {
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "VmRSS:") {
			if fields := strings.Fields(line); len(fields) >= 2 {
				kb, _ := strconv.ParseInt(fields[1], 10, 64)
				return kb << 10
			}
		}
	}
	return 0
}
