package llm

import (
	"encoding/json"
	"errors"
	"io/fs"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/xgrammar"
)

// grammarJSONWhitespaceCases are what format "json"'s GBNF must accept and
// refuse. llama-server's own JSON-schema converter allows, in one gap between
// tokens, nothing, one space, or one or two newlines and up to 20 spaces or
// tabs (common/json-schema-to-grammar.cpp, SPACE_RULE); format "json" is the
// fork's own grammar and must allow no more, or a model that degenerates into
// whitespace runs to num_predict, or to llama-server's 100-repeat abort (an
// HTTP 500 with no output).
//
// Bounding the rule is not enough if two of them can sit side by side: until
// 2026-10-13 "{" ws ... ws "}" and a literal's trailing ws before ws "}" each
// let two gaps run together, 44 characters in a row. The grammar now has
// llama.cpp's grammars/json.gbnf shape (a value ends in ws, and nothing puts
// ws before "}" or "]"), so an empty "{  }" is refused there as it is here.
// gapAtBound is the longest run one gap allows: two newlines and 20 spaces.
var gapAtBound = "\n\n" + strings.Repeat(" ", 20)

var grammarJSONWhitespaceCases = []struct {
	name, in string
	accept   bool
}{
	{"compact", `{"a":1,"b":[true,null]}`, true},
	{"one space after each literal", `{ "a": 1, "b": [ true ] }`, true},
	{"newline and ten levels of two-space indent", "{\n                    \"a\": 1\n}", true},
	{"two newlines and a tab", "{\n\n\t\"a\": 1\n}", true},
	{"pretty-printed nesting", "{\n  \"a\": {\n    \"b\": [\n      1,\n      true\n    ]\n  },\n  \"c\": null\n}", true},
	{"one gap at the bound after a literal", `{"a":true` + gapAtBound + `}`, true},
	{"one gap at the bound in an empty object", `{` + gapAtBound + `}`, true},
	{"empty object with one space", `{ }`, true},
	{"empty object with a newline", "{\n}", true},
	{"empty array with one space", `{"a":[ ]}`, true},
	{"a space after a value before a comma", `{"a":1 ,"b":"x" ,"c":{} ,"d":[] }`, true},
	{"two spaces", `{  "a":1}`, false},
	{"a run of spaces", `{` + strings.Repeat(" ", 32) + `"a":1}`, false},
	{"three newlines", "{\n\n\n\"a\":1}", false},
	{"a newline and 21 spaces", "{\n" + strings.Repeat(" ", 21) + `"a":1}`, false},
	{"empty object with two spaces", `{  }`, false},
	{"empty array with two spaces", `{"a":[  ]}`, false},
	// Two gaps at the bound run together: 44 characters in a row.
	{"empty object with two gaps at the bound", `{` + gapAtBound + gapAtBound + `}`, false},
	{"empty array with two gaps at the bound", `{"a":[` + gapAtBound + gapAtBound + `]}`, false},
	{"two gaps at the bound after a literal in an object", `{"a":true` + gapAtBound + gapAtBound + `}`, false},
	{"two gaps at the bound after a literal in an array", `{"a":[null` + gapAtBound + gapAtBound + `]}`, false},
	{"two spaces after a literal", `{"a":false  }`, false},
	{"a space between a key and its colon", `{"a" :1}`, false},
	{"whitespace after the closing brace", `{"a":1} `, false},
}

// TestGrammarJSONBoundsWhitespace checks grammarJSON against
// grammarJSONWhitespaceCases with the reference GBNF recognizer, in plain Go,
// so it runs wherever the tests do.
func TestGrammarJSONBoundsWhitespace(t *testing.T) {
	g, err := parseGBNF(grammarJSON)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range grammarJSONWhitespaceCases {
		t.Run(c.name, func(t *testing.T) {
			if got := g.accepts(c.in); got != c.accept {
				t.Errorf("accepted = %v, want %v for %q", got, c.accept, c.in)
			}
		})
	}
}

// TestGrammarJSONBoundsWhitespaceXGrammar drives the same cases through a
// real grammar engine one byte at a time, as a decoder would be masked.
// llama.cpp's grammar engine has no Go entry point, so this is xgrammar's,
// which reads the same GBNF; it runs wherever the MLX payload is built and
// skips elsewhere.
func TestGrammarJSONBoundsWhitespaceXGrammar(t *testing.T) {
	compiler := byteGrammarCompiler(t)
	for _, c := range grammarJSONWhitespaceCases {
		t.Run(c.name, func(t *testing.T) {
			got := acceptsBytes(t, compiler, grammarJSON, c.in)
			if got != c.accept {
				t.Errorf("accepted = %v, want %v for %q", got, c.accept, c.in)
			}
		})
	}
}

const (
	grammarTestVocab       = 257
	grammarTestEOS   int32 = 256
)

func byteGrammarCompiler(t *testing.T) *xgrammar.Compiler {
	t.Helper()
	path, err := mlx.LoadedLibraryPath()
	if err != nil {
		t.Skipf("native MLX payload is not built: %v", err)
	}
	pieces := make([]string, grammarTestVocab)
	for i := range 256 {
		pieces[i] = string([]byte{byte(i)})
	}
	pieces[grammarTestEOS] = "<eos>"
	compiler, err := xgrammar.New(filepath.Dir(path), pieces, grammarTestVocab, []int32{grammarTestEOS}, 8, 128<<20)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			t.Skipf("native xgrammar payload is not built: %v", err)
		}
		t.Fatal(err)
	}
	t.Cleanup(compiler.Close)
	return compiler
}

// acceptsBytes reports whether grammar accepts in, fed one byte per token
// under the mask as a decoder would be, and can end there.
func acceptsBytes(t *testing.T, compiler *xgrammar.Compiler, grammar, in string) bool {
	t.Helper()
	g, err := json.Marshal(grammar)
	if err != nil {
		t.Fatal(err)
	}
	matcher, err := compiler.Compile(`{"type":"structural_tag","format":{"type":"grammar","grammar":` + string(g) + `}}`)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer matcher.Close()
	row := make([]int32, (grammarTestVocab+31)/32)
	allowed := func(id int32) bool { return uint32(row[id/32])&(uint32(1)<<uint(id%32)) != 0 }
	for i := range len(in) {
		constrained, err := matcher.Fill(row)
		if err != nil {
			t.Fatal(err)
		}
		if constrained && !allowed(int32(in[i])) {
			return false
		}
		if err := matcher.Accept(int32(in[i])); err != nil {
			return false
		}
	}
	if matcher.Terminated() {
		return true
	}
	constrained, err := matcher.Fill(row)
	if err != nil {
		t.Fatal(err)
	}
	return !constrained || allowed(grammarTestEOS)
}
