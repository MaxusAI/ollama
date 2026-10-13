package mlxrunner

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// An explicit num_ctx reaches the runner as --ctx-size; the automatic
// VRAM-tier rung does not, so the runner keeps the model's own maximum and the
// rung stays the advisory value it was before (ADR 0050).
func TestRunnerArgsCarryOnlyAnExplicitNumCtx(t *testing.T) {
	for _, tt := range []struct {
		name       string
		numCtx     int
		numCtxAuto bool
		want       []string
	}{
		{
			name: "automatic rung stays advisory", numCtx: 32768, numCtxAuto: true,
			want: []string{"runner", "--model", "m:latest", "--port", "49152"},
		},
		{
			name: "explicit num_ctx is enforced", numCtx: 8192,
			want: []string{"runner", "--model", "m:latest", "--port", "49152", "--ctx-size", "8192"},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := &Client{modelName: "m:latest", numCtxAuto: tt.numCtxAuto}
			c.softContextLength.Store(int64(tt.numCtx))
			if got := c.runnerArgs(49152); !slices.Equal(got, tt.want) {
				t.Fatalf("runnerArgs = %q, want %q", got, tt.want)
			}
			if got, want := c.EnforcesContextLength(), !tt.numCtxAuto; got != want {
				t.Fatalf("EnforcesContextLength = %v, want %v", got, want)
			}
		})
	}
}

func TestServedContextLength(t *testing.T) {
	for _, tt := range []struct {
		name                   string
		numCtx, maxCtx, served int
	}{
		{name: "no --ctx-size serves the model maximum", maxCtx: 262144, served: 262144},
		{name: "a smaller num_ctx is the window", numCtx: 8192, maxCtx: 262144, served: 8192},
		{name: "the model maximum caps num_ctx", numCtx: 524288, maxCtx: 262144, served: 262144},
		{name: "an unknown maximum leaves num_ctx", numCtx: 8192, served: 8192},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := servedContextLength(tt.numCtx, tt.maxCtx); got != tt.served {
				t.Fatalf("servedContextLength(%d, %d) = %d, want %d", tt.numCtx, tt.maxCtx, got, tt.served)
			}
		})
	}
}

func seq(n int) []int32 {
	s := make([]int32, n)
	for i := range s {
		s[i] = int32(i)
	}
	return s
}

// Under an explicit num_ctx an over-long prompt is cut exactly the way
// llama-server's path cuts it (llm completionPromptForRequest): keep num_keep
// tokens (plus BOS), drop from just after them, and leave the tail at
// llm.ContextShiftPromptLimit so generation has room.
func TestPromptWindowFitTruncatesLikeLlamaServer(t *testing.T) {
	w := promptWindow{length: 64, numCtx: true, addBOS: true}
	req := CompletionRequest{Truncate: true, Shift: true, Options: api.Options{NumKeep: 4}}

	got, err := w.fit(seq(100), false, req)
	if err != nil {
		t.Fatal(err)
	}
	keep := 4 + 1 // num_keep does not count BOS
	limit := llm.ContextShiftPromptLimit(64, keep)
	if len(got) != limit {
		t.Fatalf("truncated to %d tokens, want %d", len(got), limit)
	}
	if !slices.Equal(got[:keep], seq(keep)) {
		t.Fatalf("head = %v, want the first %d tokens", got[:keep], keep)
	}
	if got[len(got)-1] != 99 || got[keep] != int32(100-(limit-keep)) {
		t.Fatalf("tail = %v, want the last %d tokens", got[keep:], limit-keep)
	}

	// num_keep < 0 keeps as much of the head as fits, as on llama-server.
	req.Options.NumKeep = -1
	got, err = w.fit(seq(100), false, req)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, seq(63)) {
		t.Fatalf("num_keep -1 kept %v, want the first 63 tokens", got)
	}
}

func TestPromptWindowFitLeavesAPromptThatFits(t *testing.T) {
	w := promptWindow{length: 64, numCtx: true}
	got, err := w.fit(seq(63), false, CompletionRequest{})
	if err != nil || len(got) != 63 {
		t.Fatalf("fit(63 of 64) = %d tokens, %v; want the prompt unchanged", len(got), err)
	}
}

// Where llama-server's path would not truncate either, the prompt is refused
// with both lengths named; never served past the window.
func TestPromptWindowFitRefusesWhatItCannotTruncate(t *testing.T) {
	for _, tt := range []struct {
		name  string
		w     promptWindow
		media bool
		req   CompletionRequest
		want  string
	}{
		{
			name: "truncate false",
			w:    promptWindow{length: 64, numCtx: true},
			req:  CompletionRequest{Shift: true},
			want: "input length (100 tokens) exceeds the context length (64 tokens) set by num_ctx",
		},
		{
			name: "shift false",
			w:    promptWindow{length: 64, numCtx: true},
			req:  CompletionRequest{Truncate: true},
			want: "input length (100 tokens) exceeds the context length (64 tokens) set by num_ctx",
		},
		{
			name:  "media",
			w:     promptWindow{length: 64, numCtx: true},
			media: true,
			req:   CompletionRequest{Truncate: true, Shift: true},
			want:  "input length (100 tokens) exceeds the context length (64 tokens) set by num_ctx",
		},
		{
			// An explicit num_ctx above the model's maximum is capped there, so
			// the limit is the model's, not num_ctx's: a client must not read it
			// as a rung a larger num_ctx would cure.
			name: "num_ctx above the model maximum",
			w:    promptWindow{length: 64, numCtx: true, requested: 128},
			req:  CompletionRequest{Shift: true},
			want: "input length (100 tokens) exceeds the model's maximum context length (64 tokens); num_ctx 128 is larger than the model's maximum",
		},
		{
			// No explicit num_ctx: the window is the model's maximum and an
			// over-long prompt is refused, as upstream does.
			name: "model maximum",
			w:    promptWindow{length: 64},
			req:  CompletionRequest{Truncate: true, Shift: true},
			want: "input length (100 tokens) exceeds the model's maximum context length (64 tokens)",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.w.fit(seq(100), tt.media, tt.req)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("fit error = %v, want it to contain %q", err, tt.want)
			}
		})
	}
}

// The runner needs the request's truncate and shift to decide between cutting
// and refusing, so both ride the wire.
func TestCompletionForwardsTruncateAndShift(t *testing.T) {
	var got CompletionRequest
	c := newCompletionTestClient(func(r *http.Request) string {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode wire request: %v", err)
		}
		return `{"Content":"ok","Done":true}` + "\n"
	})
	if err := c.Completion(context.Background(), llm.CompletionRequest{Prompt: "hi", Truncate: true, Shift: true}, func(llm.CompletionResponse) {}); err != nil {
		t.Fatal(err)
	}
	if !got.Truncate || !got.Shift {
		t.Fatalf("wire Truncate=%v Shift=%v, want both true", got.Truncate, got.Shift)
	}
}

// What the client reports, and /api/ps shows, is the window the runner
// enforces when num_ctx is explicit: the runner's own status, which
// servedContextLength has already capped at the model's maximum.
func TestPingReportsTheEnforcedWindow(t *testing.T) {
	for _, tt := range []struct {
		name           string
		numCtx, status int
		numCtxAuto     bool
		want           int
	}{
		{name: "explicit", numCtx: 8192, status: 8192, want: 8192},
		{name: "explicit above the model maximum", numCtx: 524288, status: 262144, want: 262144},
		{name: "automatic stays the advisory rung", numCtx: 32768, status: 262144, numCtxAuto: true, want: 32768},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := newCompletionTestClient(func(*http.Request) string {
				b, _ := json.Marshal(statusResponse{ContextLength: tt.status})
				return string(b)
			})
			c.numCtxAuto = tt.numCtxAuto
			c.softContextLength.Store(int64(tt.numCtx))
			if err := c.Ping(context.Background()); err != nil {
				t.Fatal(err)
			}
			if got := c.ContextLength(); got != tt.want {
				t.Fatalf("ContextLength = %d, want %d", got, tt.want)
			}
		})
	}
}
