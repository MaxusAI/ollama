package mlxrunner

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestCompletionForwardsFormatAsStructuralTag(t *testing.T) {
	var got CompletionRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode wire request: %v", err)
		}
		json.NewEncoder(w).Encode(CompletionResponse{Done: true}) //nolint:errcheck
	}))
	defer srv.Close()

	c := &Client{
		port:   srv.Listener.Addr().(*net.TCPAddr).Port,
		client: http.DefaultClient,
		status: llm.NewStatusWriter(io.Discard),
	}
	err := c.Completion(context.Background(), llm.CompletionRequest{
		Prompt:  "p",
		Format:  json.RawMessage(`{"type":"object"}`),
		Options: &api.Options{},
	}, func(llm.CompletionResponse) {})
	if err != nil {
		t.Fatalf("Completion: %v", err)
	}
	// Since v0.34.0 structured output is compiled as an xgrammar structural
	// tag: the request's schema must reach the runner inside one.
	want := `{"type":"structural_tag","format":{"type":"json_schema","max_whitespace_cnt":32,"json_schema":{"type":"object"}}}`
	if string(got.Format) != want {
		t.Errorf("wire Format = %q, want the request's schema forwarded as %s", got.Format, want)
	}
}

func TestFormatNeverSilentlyDropsAConstraint(t *testing.T) {
	// ADR 0009's guarantee, re-asserted against the v0.34.0 grammar path. The
	// client wraps every non-empty format in a structural tag (requestGrammar)
	// and the runner parses only structural tags (parseGrammar), so a format
	// must either reach the grammar compiler or be an error. It may never
	// come out of this path as "no constraint". Rejecting a format the
	// compiler cannot honour, such as "yaml", now happens in the native
	// xgrammar compile, which needs the MLX payload; the fold's GPU gate
	// checks that one live.

	// Wire values, not Go strings: an absent format decodes to a zero-length
	// RawMessage, while "format":"" decodes to the two bytes `""`.
	for _, c := range []struct {
		name   string
		format json.RawMessage
	}{
		{name: "absent", format: nil},
		{name: "null", format: json.RawMessage(`null`)},
		{name: "empty string", format: json.RawMessage(`""`)},
	} {
		spec, err := parseGrammar(requestGrammar(llm.CompletionRequest{Format: c.format}))
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
		}
		if spec != "" {
			t.Errorf("%s: unexpected constraint %q", c.name, spec)
		}
	}

	for _, c := range []struct {
		name   string
		format json.RawMessage
	}{
		{name: "json", format: json.RawMessage(`"json"`)},
		{name: "schema", format: json.RawMessage(`{"type":"object","properties":{"a":{"type":"string"}}}`)},
		{name: "unknown string", format: json.RawMessage(`"yaml"`)},
	} {
		spec, err := parseGrammar(requestGrammar(llm.CompletionRequest{Format: c.format}))
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if spec == "" {
			t.Errorf("%s: the constraint was dropped", c.name)
		}
	}

	if _, err := parseGrammar(requestGrammar(llm.CompletionRequest{Format: json.RawMessage(`{"type":`)})); err == nil {
		t.Fatal("malformed schema: expected an error")
	}
	// A caller that skips requestGrammar gets an error, not an unconstrained request.
	for _, raw := range []string{`"json"`, `{"type":"object"}`, `"yaml"`} {
		if _, err := parseGrammar(json.RawMessage(raw)); err == nil {
			t.Errorf("parseGrammar(%s): expected an error for an unwrapped format", raw)
		}
	}
}

func TestStructuralTagBoundsTheWhitespaceBetweenTokens(t *testing.T) {
	// Without max_whitespace_cnt xgrammar compiles every separator to "[ \n\t]*",
	// so a stalled model can spend a whole generation budget on indentation and
	// return an answer that is never closed. The bound is what ends that run, and
	// it belongs on the json_schema element itself, which a thinking response
	// nests behind its closing string.
	for _, tt := range []struct {
		name string
		req  llm.CompletionRequest
	}{
		{name: "format", req: llm.CompletionRequest{Format: json.RawMessage(`{"type":"object"}`)}},
		{name: "format after thinking", req: llm.CompletionRequest{Format: json.RawMessage(`{"type":"object"}`), ThinkingClose: []string{"</think>"}}},
		{name: "json after thinking with two closings", req: llm.CompletionRequest{Format: json.RawMessage(`"json"`), ThinkingClose: []string{"<channel|>", "</think>"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			raw := requestGrammar(tt.req)
			var tag any
			if err := json.Unmarshal(raw, &tag); err != nil {
				t.Fatalf("structural tag is not valid JSON: %v (%s)", err, raw)
			}
			elements := jsonSchemaElements(tag)
			if len(elements) != 1 {
				t.Fatalf("structural tag has %d json_schema elements, want 1: %s", len(elements), raw)
			}
			bound, ok := elements[0]["max_whitespace_cnt"].(float64)
			if !ok {
				t.Fatalf("json_schema element leaves max_whitespace_cnt unset: whitespace is unbounded again (%s)", raw)
			}
			if bound <= 0 {
				t.Errorf("max_whitespace_cnt = %v, want a positive bound", bound)
			}
		})
	}
}

// jsonSchemaElements returns every json_schema element in a decoded structural tag.
func jsonSchemaElements(v any) []map[string]any {
	var found []map[string]any
	switch v := v.(type) {
	case map[string]any:
		if v["type"] == "json_schema" {
			found = append(found, v)
		}
		for _, child := range v {
			found = append(found, jsonSchemaElements(child)...)
		}
	case []any:
		for _, child := range v {
			found = append(found, jsonSchemaElements(child)...)
		}
	}
	return found
}

func TestBoundedWhitespaceKeepsTheSchemaIntact(t *testing.T) {
	// The bound is added beside the schema, never inside it.
	schema := `{"type":"object","properties":{"a":{"type":"string"}}}`
	var tag struct {
		Format struct {
			JSONSchema json.RawMessage `json:"json_schema"`
		} `json:"format"`
	}
	raw := requestGrammar(llm.CompletionRequest{Format: json.RawMessage(schema)})
	if err := json.Unmarshal(raw, &tag); err != nil {
		t.Fatalf("structural tag is not valid JSON: %v (%s)", err, raw)
	}
	if string(tag.Format.JSONSchema) != schema {
		t.Errorf("json_schema = %s, want the request's schema unchanged %s", tag.Format.JSONSchema, schema)
	}
}
