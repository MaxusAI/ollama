package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestCompletionForwardsFormat(t *testing.T) {
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
	if string(got.Format) != `{"type":"object"}` {
		t.Errorf("wire Format = %q, want the request's format forwarded", got.Format)
	}
}

func TestCompletionRejectsRawGrammar(t *testing.T) {
	c := &Client{client: http.DefaultClient, status: llm.NewStatusWriter(io.Discard)}
	err := c.Completion(context.Background(), llm.CompletionRequest{
		Prompt:  "p",
		Grammar: `root ::= "x"`,
		Options: &api.Options{},
	}, func(llm.CompletionResponse) {})
	var se api.StatusError
	if !errors.As(err, &se) {
		t.Fatalf("Completion with Grammar: err = %v, want api.StatusError", err)
	}
	if se.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", se.StatusCode)
	}
}

func TestRequestCompileFormat(t *testing.T) {
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
		req := &Request{CompletionRequest: CompletionRequest{Format: c.format}}
		if err := req.compileFormat(); err != nil {
			t.Errorf("compileFormat(%s): %v", c.name, err)
		}
		if req.Constraint != nil {
			t.Errorf("compileFormat(%s): unexpected constraint", c.name)
		}
	}

	req := &Request{CompletionRequest: CompletionRequest{Format: json.RawMessage(`"json"`)}}
	if err := req.compileFormat(); err != nil {
		t.Fatalf("compileFormat(json): %v", err)
	}
	if req.Constraint == nil {
		t.Fatal("compileFormat(json): no constraint")
	}

	req = &Request{CompletionRequest: CompletionRequest{Format: json.RawMessage(`"yaml"`)}}
	if err := req.compileFormat(); err == nil {
		t.Fatal("compileFormat(yaml): expected error")
	}
}

// TestPrefillChunkLen walks a prompt the way prefill does and checks what
// media chunking must hold: the chunks tile the prompt up to the seed token,
// no boundary lands inside an image block, and every block is carried whole
// by a chunk starting at position zero.
func TestPrefillChunkLen(t *testing.T) {
	const chunk = 8
	for _, tc := range []struct {
		name  string
		total int
		from  int        // resume position; 0 for a full prefill
		spans [][2]int32 // image soft-token blocks, [start, end)
		want  []int      // expected chunk lengths
	}{
		{name: "text only", total: 20, want: []int{8, 8, 3}},
		{name: "block inside the opening chunk", total: 30, spans: [][2]int32{{2, 6}}, want: []int{8, 8, 8, 5}},
		{name: "block past the nominal chunk", total: 40, spans: [][2]int32{{11, 19}}, want: []int{19, 8, 8, 4}},
		{name: "block longer than a chunk", total: 60, spans: [][2]int32{{3, 40}}, want: []int{40, 8, 8, 3}},
		{name: "two blocks", total: 64, spans: [][2]int32{{2, 9}, {21, 33}}, want: []int{33, 8, 8, 8, 6}},
		{name: "block ends at the seed", total: 12, spans: [][2]int32{{2, 11}}, want: []int{11}},
		{name: "resumed before a block", total: 40, from: 16, spans: [][2]int32{{20, 26}}, want: []int{4, 8, 8, 3}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var got []int
			covered := make([]int, len(tc.spans))
			for position := tc.from; tc.total-position > 1; {
				limit := tc.total - position - 1
				n := prefillChunkLen(position, limit, chunk, tc.spans)
				if n < 1 || n > limit {
					t.Fatalf("position %d: chunk of %d tokens, want 1..%d", position, n, limit)
				}
				got = append(got, n)

				end := position + n
				for i, s := range tc.spans {
					start, blockEnd := int(s[0]), int(s[1])
					if start >= end || blockEnd <= position {
						continue // this chunk does not touch the block
					}
					if start < position || blockEnd > end {
						t.Errorf("chunk [%d,%d) splits image block %v", position, end, s)
						continue
					}
					// The bidirectional overlay only composes from position
					// zero, so a prefill from the start of the prompt has to
					// carry every block in its opening chunk.
					if position != 0 && tc.from == 0 {
						t.Errorf("chunk carrying image block %v starts at %d, want 0", s, position)
					}
					covered[i]++
				}
				position = end
			}

			if !slices.Equal(got, tc.want) {
				t.Errorf("chunk lengths = %v, want %v", got, tc.want)
			}
			for i, n := range covered {
				if n != 1 {
					t.Errorf("image block %v covered by %d chunks, want exactly 1", tc.spans[i], n)
				}
			}
		})
	}
}

func TestCheckVisionPrefillBudget(t *testing.T) {
	// The opening chunk must span every image block, and gemma4 masks it
	// densely at 8 bytes per cell, so the ceiling is sqrt(budget/8).
	limit := int32(11585) // floor(sqrt((1<<30)/8))

	for _, tc := range []struct {
		name    string
		spans   [][2]int32
		wantErr bool
	}{
		{"text only", nil, false},
		{"image at the front", [][2]int32{{1, 257}}, false},
		{"image just inside the ceiling", [][2]int32{{limit - 256, limit}}, false},
		{"image just past the ceiling", [][2]int32{{limit, limit + 1}}, true},
		{"long text then a late image", [][2]int32{{32000, 32256}}, true},
		{"last of several blocks decides", [][2]int32{{1, 257}, {40000, 40256}}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := checkVisionPrefillBudget(tc.spans)
			if gotErr := err != nil; gotErr != tc.wantErr {
				t.Fatalf("checkVisionPrefillBudget(%v) error = %v, want error = %v", tc.spans, err, tc.wantErr)
			}
		})
	}
}
