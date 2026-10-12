package server

import (
	"context"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
)

// mlxNumCtxRunner is an MLX runner as the routes see it: it reports whether it
// holds prompts to the num_ctx it was loaded with (ADR 0050).
type mlxNumCtxRunner struct {
	*mockRunner
	enforces bool
}

func (r mlxNumCtxRunner) EnforcesContextLength() bool { return r.enforces }

func TestPromptTruncation(t *testing.T) {
	gguf := &Model{}
	mlx := &Model{Config: model.ConfigV2{ModelFormat: "safetensors"}}
	off := false
	on := true
	explicit := mlxNumCtxRunner{mockRunner: &mockRunner{}, enforces: true}
	automatic := mlxNumCtxRunner{mockRunner: &mockRunner{}}

	for _, tt := range []struct {
		name     string
		m        *Model
		r        llm.LlamaServer
		truncate *bool
		want     bool
	}{
		{"gguf default", gguf, &mockRunner{}, nil, true},
		{"gguf truncate false", gguf, &mockRunner{}, &off, false},
		{"mlx automatic num_ctx is advisory", mlx, automatic, nil, false},
		{"mlx automatic ignores truncate true", mlx, automatic, &on, false},
		{"mlx explicit num_ctx truncates", mlx, explicit, nil, true},
		{"mlx explicit honours truncate false", mlx, explicit, &off, false},
		{"mlx runner that cannot say", mlx, &mockRunner{}, nil, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := promptTruncation(tt.truncate, tt.m, tt.r); got != tt.want {
				t.Fatalf("promptTruncation = %v, want %v", got, tt.want)
			}
		})
	}
}

func newMLXNumCtxServer(t *testing.T, enforces bool) (*Server, *mockRunner) {
	t.Helper()
	mock := &mockRunner{
		contextLength: 8,
		CompletionResponse: llm.CompletionResponse{
			Content: "ok", Done: true, DoneReason: llm.DoneReasonStop,
		},
	}
	s := newServerWithMockRunner(t, mock)
	s.sched.loadFn = func(req *LlmRequest, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) bool {
		req.successCh <- &runnerRef{llama: mlxNumCtxRunner{mockRunner: mock, enforces: enforces}, sessionDuration: time.Hour}
		return false
	}
	createSafetensorsTestModel(t, "mlx-numctx", model.ConfigV2{
		ModelFormat: "safetensors", Renderer: "qwen3.5", Capabilities: []string{"completion"},
	}, nil)
	return s, mock
}

// With an explicit num_ctx an MLX chat drops the oldest turns to fit, as the
// GGUF path does, and asks the runner to truncate what still does not fit.
// With the automatic rung nothing is dropped: that rung is advisory and the
// runner serves up to the model's maximum, as before.
func TestChatMLXExplicitNumCtxTruncatesMessages(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "")

	msgs := []api.Message{
		{Role: "user", Content: "oldest one two three four five six seven eight nine ten"},
		{Role: "user", Content: "newest"},
	}
	stream := false
	off := false

	for _, tt := range []struct {
		name         string
		enforces     bool
		options      map[string]any
		truncate     *bool
		wantOldest   bool
		wantTruncate bool
	}{
		{name: "explicit", enforces: true, options: map[string]any{"num_ctx": 8}, wantTruncate: true},
		{name: "explicit truncate false", enforces: true, options: map[string]any{"num_ctx": 8}, truncate: &off, wantOldest: true},
		{name: "automatic", wantOldest: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s, mock := newMLXNumCtxServer(t, tt.enforces)
			w := createRequest(t, s.ChatHandler, api.ChatRequest{
				Model: "mlx-numctx", Messages: msgs, Stream: &stream, Options: tt.options, Truncate: tt.truncate,
			})
			if w.Code != http.StatusOK {
				t.Fatalf("status %d: %s", w.Code, w.Body)
			}
			prompt := mock.CompletionRequest.Prompt
			if got := strings.Contains(prompt, "oldest"); got != tt.wantOldest {
				t.Fatalf("oldest turn in prompt = %v, want %v; prompt %q", got, tt.wantOldest, prompt)
			}
			if !strings.Contains(prompt, "newest") {
				t.Fatalf("latest turn dropped: %q", prompt)
			}
			if mock.CompletionRequest.Truncate != tt.wantTruncate {
				t.Fatalf("runner Truncate = %v, want %v", mock.CompletionRequest.Truncate, tt.wantTruncate)
			}
		})
	}
}

// /api/generate hands the runner the same decision: truncate under an explicit
// num_ctx, refuse past the model maximum under the automatic rung.
func TestGenerateMLXExplicitNumCtxTruncates(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "")
	stream := false

	for _, tt := range []struct {
		name     string
		enforces bool
		raw      bool
		want     bool
	}{
		{name: "explicit", enforces: true, want: true},
		{name: "explicit raw", enforces: true, raw: true, want: true},
		{name: "automatic"},
		{name: "automatic raw", raw: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s, mock := newMLXNumCtxServer(t, tt.enforces)
			w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
				Model: "mlx-numctx", Prompt: "hello", Raw: tt.raw, Stream: &stream,
			})
			if w.Code != http.StatusOK {
				t.Fatalf("status %d: %s", w.Code, w.Body)
			}
			if mock.CompletionRequest.Truncate != tt.want {
				t.Fatalf("runner Truncate = %v, want %v", mock.CompletionRequest.Truncate, tt.want)
			}
		})
	}
}

// An explicit num_ctx arriving at an MLX runner loaded on the automatic rung
// reloads it even when the numbers match: that runner was started without the
// limit, so serving the request from it would not enforce it. The reverse does
// not reload, so alternating clients cannot churn the runner.
func TestSchedNeedsReloadMLXExplicitNumCtxOverAutomatic(t *testing.T) {
	ctx, done := context.WithTimeout(t.Context(), schedTestTimeout(100*time.Millisecond))
	defer done()

	opts := api.DefaultOptions()
	opts.NumCtx = 32768
	mlxModel := &Model{Config: model.ConfigV2{ModelFormat: "safetensors"}}
	runner := &runnerRef{
		model:       mlxModel,
		Options:     &opts,
		llama:       &mockLlm{vramByGPU: map[ml.DeviceID]uint64{}},
		numParallel: 1,
		numCtxAuto:  true,
	}
	req := &LlmRequest{model: mlxModel, opts: api.DefaultOptions()}
	req.opts.NumCtx = 32768

	require.True(t, runner.needsReload(ctx, req), "an explicit num_ctx must reach a runner that enforces it")

	runner.numCtxAuto = false
	req.numCtxAuto = true
	require.False(t, runner.needsReload(ctx, req), "an enforcing runner serves the automatic rung of the same size")
}
