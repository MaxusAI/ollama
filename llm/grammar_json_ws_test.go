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

// TestGrammarJSONBoundsWhitespace drives format "json"'s GBNF through a
// grammar engine one byte at a time. llama-server's own JSON-schema converter
// allows, between tokens, nothing, one space, or one or two newlines and up to
// 20 spaces or tabs (common/json-schema-to-grammar.cpp, SPACE_RULE); format
// "json" is the fork's own grammar and must allow no more, or a model that
// degenerates into whitespace runs to num_predict, or to llama-server's
// 100-repeat abort (an HTTP 500 with no output).
//
// llama.cpp's grammar engine has no Go entry point, so the grammar is checked
// by xgrammar's, which reads the same GBNF; it runs wherever the MLX payload
// is built and skips elsewhere.
func TestGrammarJSONBoundsWhitespace(t *testing.T) {
	compiler := byteGrammarCompiler(t)
	for _, c := range []struct {
		name, in string
		accept   bool
	}{
		{"compact", `{"a":1,"b":[true,null]}`, true},
		{"one space after each literal", `{ "a": 1, "b": [ true ] }`, true},
		{"newline and ten levels of two-space indent", "{\n                    \"a\": 1\n}", true},
		{"two newlines and a tab", "{\n\n\t\"a\": 1\n}", true},
		{"two spaces", `{  "a":1}`, false},
		{"a run of spaces", `{` + strings.Repeat(" ", 32) + `"a":1}`, false},
		{"three newlines", "{\n\n\n\"a\":1}", false},
		{"a newline and 21 spaces", "{\n" + strings.Repeat(" ", 21) + `"a":1}`, false},
	} {
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
