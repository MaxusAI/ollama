package mlxrunner

import (
	"encoding/json"
	"errors"
	"io/fs"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlxrunner/xgrammar"
)

// TestRequestGrammarWhitespaceRunMatchesLlamaCpp drives the structural tag the
// runner decodes under through xgrammar one byte at a time (ADR 0035 checked the
// bound only on the tag's JSON). The longest whitespace run between tokens is
// llama.cpp's for a JSON schema on the GGUF path: two newlines and 20 spaces or
// tabs, 22 characters, which still fits a newline and ten levels of two-space
// indent. muse-glimmer answering after its message header pads the first and
// last slot to the bound in single pass (2026-10-12), so the bound is also what
// that padding costs.
func TestRequestGrammarWhitespaceRunMatchesLlamaCpp(t *testing.T) {
	compiler := byteXGrammar(t)
	schema := json.RawMessage(`{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}`)
	for _, tt := range []struct {
		name, prefix string
		req          llm.CompletionRequest
	}{
		{name: "format", req: llm.CompletionRequest{Format: schema}},
		{name: "format after thinking", prefix: "plan</think>", req: llm.CompletionRequest{Format: schema, ThinkingClose: []string{"</think>"}}},
		{name: "json after a header", prefix: " to=user<|message|>", req: llm.CompletionRequest{Format: json.RawMessage(`"json"`), ThinkingClose: []string{" to=user<|message|>"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			tag := string(requestGrammar(tt.req))
			for _, c := range []struct {
				name, in string
				accept   bool
			}{
				{"compact", `{"a":1}`, true},
				{"22 spaces", `{` + strings.Repeat(" ", 22) + `"a":1}`, true},
				{"23 spaces", `{` + strings.Repeat(" ", 23) + `"a":1}`, false},
				{"two newlines and 20 spaces", "{\n\n" + strings.Repeat(" ", 20) + `"a":1}`, true},
				{"a newline and ten levels of indent", "{\n" + strings.Repeat("  ", 10) + "\"a\": 1\n}", true},
			} {
				if got := xgrammarAccepts(t, compiler, tag, tt.prefix+c.in); got != c.accept {
					t.Errorf("%s: accepted = %v, want %v", c.name, got, c.accept)
				}
			}
		})
	}
}

const (
	byteVocab       = 257
	byteEOS   int32 = 256
)

func byteXGrammar(t *testing.T) *xgrammar.Compiler {
	t.Helper()
	path, err := mlx.LoadedLibraryPath()
	if err != nil {
		t.Skipf("native MLX payload is not built: %v", err)
	}
	pieces := make([]string, byteVocab)
	for i := range 256 {
		pieces[i] = string([]byte{byte(i)})
	}
	pieces[byteEOS] = "<eos>"
	compiler, err := xgrammar.New(filepath.Dir(path), pieces, byteVocab, []int32{byteEOS}, 8, 128<<20)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			t.Skipf("native xgrammar payload is not built: %v", err)
		}
		t.Fatal(err)
	}
	t.Cleanup(compiler.Close)
	return compiler
}

// xgrammarAccepts reports whether tag accepts in, fed one byte per token under
// the mask as a decoder would be, and can end there.
func xgrammarAccepts(t *testing.T, compiler *xgrammar.Compiler, tag, in string) bool {
	t.Helper()
	matcher, err := compiler.Compile(tag)
	if err != nil {
		t.Fatalf("compile %s: %v", tag, err)
	}
	defer matcher.Close()
	row := make([]int32, (byteVocab+31)/32)
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
	return !constrained || allowed(byteEOS)
}
