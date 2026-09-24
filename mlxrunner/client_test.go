package mlxrunner

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/ollama/ollama/llm"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return fn(req) }

func newCompletionTestClient(handler func(*http.Request) string) *Client {
	return &Client{
		port:   11434,
		status: llm.NewStatusWriter(io.Discard),
		client: &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(strings.NewReader(handler(req))),
				Request:    req,
			}, nil
		})},
	}
}

// Media must ride the wire to the runner subprocess: the runner owns
// model-specific preprocessing and the does-this-model-support-images check,
// so the client forwards payloads instead of judging them.
func TestCompletionForwardsMedia(t *testing.T) {
	img := []byte{0x89, 'P', 'N', 'G', 1, 2, 3}
	var got CompletionRequest
	c := newCompletionTestClient(func(r *http.Request) string {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode wire request: %v", err)
		}
		return `{"Content":"ok","Done":true}` + "\n"
	})

	err := c.Completion(context.Background(), llm.CompletionRequest{
		Prompt: "describe [img-0]",
		Media:  []llm.MediaData{{Data: img, ID: 0, Kind: llm.MediaKindImage}},
	}, func(llm.CompletionResponse) {})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Media) != 1 {
		t.Fatalf("expected 1 media entry on the wire, got %d", len(got.Media))
	}
	if !bytes.Equal(got.Media[0].Data, img) || got.Media[0].ID != 0 || got.Media[0].Kind != llm.MediaKindImage {
		t.Fatalf("media not forwarded intact: %+v", got.Media[0])
	}
}

func testIntPtr(v int) *int {
	return &v
}

func TestRequestGrammar(t *testing.T) {
	schema := `{"type":"object","properties":{"answer":{"type":"string"}}}`
	// The tag carries max_whitespace_cnt so a stalled decode cannot spend the whole
	// generation budget on indentation; see maxWhitespaceRun.
	tag := `{"type":"structural_tag","format":{"type":"json_schema","max_whitespace_cnt":32,"json_schema":` + schema + `}}`
	for _, tt := range []struct {
		name string
		req  llm.CompletionRequest
		want string
	}{
		{name: "unset"},
		{name: "null", req: llm.CompletionRequest{Format: json.RawMessage(`null`)}},
		{name: "empty", req: llm.CompletionRequest{Format: json.RawMessage(`""`)}},
		{
			name: "json",
			req:  llm.CompletionRequest{Format: json.RawMessage(`"json"`)},
			want: `{"type":"structural_tag","format":{"type":"json_schema","max_whitespace_cnt":32,"json_schema":{"type":"object"}}}`,
		},
		{name: "schema", req: llm.CompletionRequest{Format: json.RawMessage(schema)}, want: tag},
		{
			name: "schema after thinking",
			req:  llm.CompletionRequest{Format: json.RawMessage(schema), ThinkingClose: []string{"</think>"}},
			want: `{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"any_text","excludes":["</think>"]},` +
				`{"type":"optional","content":{"type":"sequence","elements":[{"type":"const_string","value":"</think>"},` +
				`{"type":"json_schema","json_schema":` + schema + `}]}}]}}`,
		},
		{
			name: "schema after thinking with two closings",
			req:  llm.CompletionRequest{Format: json.RawMessage(schema), ThinkingClose: []string{"<|final|>", "<|final|>json"}},
			want: `{"type":"structural_tag","format":{"type":"sequence","elements":[{"type":"any_text","excludes":["<|final|>","<|final|>json"]},` +
				`{"type":"optional","content":{"type":"sequence","elements":[{"type":"or","elements":[{"type":"const_string","value":"<|final|>"},{"type":"const_string","value":"<|final|>json"}]},` +
				`{"type":"json_schema","json_schema":` + schema + `}]}}]}}`,
		},
		{name: "thinking without a format", req: llm.CompletionRequest{ThinkingClose: []string{"</think>"}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := string(requestGrammar(tt.req)); got != tt.want {
				t.Fatalf("requestGrammar = %s, want %s", got, tt.want)
			}
		})
	}
}
