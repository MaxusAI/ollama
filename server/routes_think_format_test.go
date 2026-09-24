package server

import (
	"bufio"
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// Structured outputs on a thinking model are one completion call (upstream
// v0.34.4, 5a0ff3116): the format goes to the runner verbatim together with the
// strings that end the thinking, and the runner leaves the thinking free until
// one of them. Upstream tests the runners' grammars but not the handlers; these
// pin what each endpoint sends and what it returns.

// thinkFormatSchema is the format every test in this file sends.
var thinkFormatSchema = json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}`)

// thinkFormatMetrics is what the fake runner reports. The prompt count includes
// image tokens a text tokenization cannot see, so a handler that recomputes it
// instead of passing it through is caught by the value.
var thinkFormatMetrics = llm.CompletionResponse{
	PromptEvalCount:    1500,
	PromptEvalDuration: 7 * time.Millisecond,
	EvalCount:          42,
	EvalDuration:       11 * time.Millisecond,
}

type thinkFormatCase struct {
	parser   string
	think    *api.ThinkValue
	closings []string // the parser's own answer (model/parsers TestThinkingClose)
	output   string   // the model's raw output: thinking, a closing, then JSON
	thinking string
}

func thinkFormatCases() []thinkFormatCase {
	on := &api.ThinkValue{Value: true}
	return []thinkFormatCase{
		{parser: "nemotron-3-nano", think: on, closings: []string{"</think>"}, output: "Let me think.</think>" + `{"answer":"42"}`, thinking: "Let me think."},
		{parser: "qwen3.5", think: on, closings: []string{"</think>"}, output: "Let me think.</think>\n\n" + `{"answer":"42"}`, thinking: "Let me think."},
		{parser: "gemma4", think: on, closings: []string{"<channel|>"}, output: "<|channel>thought\nLet me think.<channel|>" + `{"answer":"42"}`, thinking: "Let me think."},
	}
}

// recordCompletions makes mock record every completion request and answer each
// with chunks, the last of them done with reason and the shared metrics.
func recordCompletions(mock *mockRunner, reason llm.DoneReason, chunks ...string) func() []llm.CompletionRequest {
	var (
		mu       sync.Mutex
		requests []llm.CompletionRequest
	)
	mock.CompletionFn = func(ctx context.Context, r llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
		mu.Lock()
		requests = append(requests, r)
		mu.Unlock()
		for _, c := range chunks {
			fn(llm.CompletionResponse{Content: c})
		}
		final := thinkFormatMetrics
		final.Done, final.DoneReason = true, reason
		fn(final)
		return nil
	}
	return func() []llm.CompletionRequest {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(requests)
	}
}

func checkOneFormattedCall(t *testing.T, requests []llm.CompletionRequest, wantClosings []string) {
	t.Helper()
	if len(requests) != 1 {
		t.Fatalf("got %d completion calls, want 1", len(requests))
	}
	if got := string(requests[0].Format); got != string(thinkFormatSchema) {
		t.Errorf("Format = %s, want the request's format verbatim %s", got, thinkFormatSchema)
	}
	if !slices.Equal(requests[0].ThinkingClose, wantClosings) {
		t.Errorf("ThinkingClose = %q, want %q", requests[0].ThinkingClose, wantClosings)
	}
}

func checkPassedThroughMetrics(t *testing.T, got api.Metrics) {
	t.Helper()
	if got.PromptEvalCount != thinkFormatMetrics.PromptEvalCount || got.EvalCount != thinkFormatMetrics.EvalCount ||
		got.PromptEvalDuration != thinkFormatMetrics.PromptEvalDuration || got.EvalDuration != thinkFormatMetrics.EvalDuration {
		t.Errorf("metrics = %+v, want the runner's own report %+v passed through", got, thinkFormatMetrics)
	}
}

func TestThinkFormatSinglePass(t *testing.T) {
	gin.SetMode(gin.TestMode)
	stream := false

	for _, tc := range thinkFormatCases() {
		t.Run("chat/"+tc.parser, func(t *testing.T) {
			mock := &mockRunner{}
			s := setupImplicitThinkingModel(t, mock, "think-format-chat-"+tc.parser, tc.parser)
			requests := recordCompletions(mock, llm.DoneReasonStop, tc.output)

			w := createRequest(t, s.ChatHandler, api.ChatRequest{
				Model:    "think-format-chat-" + tc.parser,
				Messages: []api.Message{{Role: "user", Content: "Answer in JSON."}},
				Think:    tc.think,
				Format:   thinkFormatSchema,
				Stream:   &stream,
			})
			if w.Code != http.StatusOK {
				t.Fatalf("status %d: %s", w.Code, w.Body.String())
			}
			checkOneFormattedCall(t, requests(), tc.closings)

			var resp api.ChatResponse
			if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
				t.Fatal(err)
			}
			if resp.Message.Thinking != tc.thinking {
				t.Errorf("thinking = %q, want %q", resp.Message.Thinking, tc.thinking)
			}
			if got := strings.TrimSpace(resp.Message.Content); got != `{"answer":"42"}` {
				t.Errorf("content = %q, want the JSON after the thinking", got)
			}
			if resp.DoneReason != "stop" {
				t.Errorf("done_reason = %q, want stop", resp.DoneReason)
			}
			checkPassedThroughMetrics(t, resp.Metrics)
		})

		t.Run("generate/"+tc.parser, func(t *testing.T) {
			mock := &mockRunner{}
			s := setupImplicitThinkingModel(t, mock, "think-format-gen-"+tc.parser, tc.parser)
			requests := recordCompletions(mock, llm.DoneReasonStop, tc.output)

			w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
				Model:  "think-format-gen-" + tc.parser,
				Prompt: "Answer in JSON.",
				Think:  tc.think,
				Format: thinkFormatSchema,
				Stream: &stream,
			})
			if w.Code != http.StatusOK {
				t.Fatalf("status %d: %s", w.Code, w.Body.String())
			}
			checkOneFormattedCall(t, requests(), tc.closings)

			var resp api.GenerateResponse
			if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
				t.Fatal(err)
			}
			if resp.Thinking != tc.thinking {
				t.Errorf("thinking = %q, want %q", resp.Thinking, tc.thinking)
			}
			if got := strings.TrimSpace(resp.Response); got != `{"answer":"42"}` {
				t.Errorf("response = %q, want the JSON after the thinking", got)
			}
			checkPassedThroughMetrics(t, resp.Metrics)
		})
	}
}

// TestThinkFormatWithoutClosings covers the requests whose response does not
// begin inside thinking. They must carry no closing strings, or the runner
// would leave the whole response free waiting for a close that never comes.
func TestThinkFormatWithoutClosings(t *testing.T) {
	gin.SetMode(gin.TestMode)
	stream := false
	off := &api.ThinkValue{Value: false}
	on := &api.ThinkValue{Value: true}

	for _, parser := range []string{"nemotron-3-nano", "qwen3.5", "gemma4"} {
		t.Run("chat think off/"+parser, func(t *testing.T) {
			mock := &mockRunner{}
			s := setupImplicitThinkingModel(t, mock, "think-off-chat-"+parser, parser)
			requests := recordCompletions(mock, llm.DoneReasonStop, `{"answer":"42"}`)
			w := createRequest(t, s.ChatHandler, api.ChatRequest{
				Model:    "think-off-chat-" + parser,
				Messages: []api.Message{{Role: "user", Content: "Answer in JSON."}},
				Think:    off,
				Format:   thinkFormatSchema,
				Stream:   &stream,
			})
			if w.Code != http.StatusOK {
				t.Fatalf("status %d: %s", w.Code, w.Body.String())
			}
			checkOneFormattedCall(t, requests(), nil)
		})

		t.Run("generate think off/"+parser, func(t *testing.T) {
			mock := &mockRunner{}
			s := setupImplicitThinkingModel(t, mock, "think-off-gen-"+parser, parser)
			requests := recordCompletions(mock, llm.DoneReasonStop, `{"answer":"42"}`)
			w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
				Model:  "think-off-gen-" + parser,
				Prompt: "Answer in JSON.",
				Think:  off,
				Format: thinkFormatSchema,
				Stream: &stream,
			})
			if w.Code != http.StatusOK {
				t.Fatalf("status %d: %s", w.Code, w.Body.String())
			}
			checkOneFormattedCall(t, requests(), nil)
		})

		t.Run("chat content prefill/"+parser, func(t *testing.T) {
			mock := &mockRunner{}
			s := setupImplicitThinkingModel(t, mock, "think-prefill-chat-"+parser, parser)
			requests := recordCompletions(mock, llm.DoneReasonStop, `"42"}`)
			w := createRequest(t, s.ChatHandler, api.ChatRequest{
				Model: "think-prefill-chat-" + parser,
				Messages: []api.Message{
					{Role: "user", Content: "Answer in JSON."},
					{Role: "assistant", Content: `{"answer":`},
				},
				Think:  on,
				Format: thinkFormatSchema,
				Stream: &stream,
			})
			if w.Code != http.StatusOK {
				t.Fatalf("status %d: %s", w.Code, w.Body.String())
			}
			checkOneFormattedCall(t, requests(), nil)
		})
	}

	// A raw prompt gives no way to tell where the response starts, so the
	// format applies from its first token.
	t.Run("generate raw", func(t *testing.T) {
		mock := &mockRunner{}
		s := setupImplicitThinkingModel(t, mock, "think-raw-gen", "qwen3.5")
		requests := recordCompletions(mock, llm.DoneReasonStop, `{"answer":"42"}`)
		w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
			Model:  "think-raw-gen",
			Prompt: "Answer in JSON.",
			Raw:    true,
			Think:  on,
			Format: thinkFormatSchema,
			Stream: &stream,
		})
		if w.Code != http.StatusOK {
			t.Fatalf("status %d: %s", w.Code, w.Body.String())
		}
		checkOneFormattedCall(t, requests(), nil)
	})
}

// TestThinkFormatStreamsThinkingBeforeContent checks that the one call's
// stream still splits into thinking chunks first and content after, and that
// only the final chunk carries metrics.
func TestThinkFormatStreamsThinkingBeforeContent(t *testing.T) {
	gin.SetMode(gin.TestMode)
	streaming := true
	mock := &mockRunner{}
	s := setupImplicitThinkingModel(t, mock, "think-format-stream", "qwen3.5")
	requests := recordCompletions(mock, llm.DoneReasonStop, "Let me ", "think.", "</think>", "\n\n", `{"answer":`, `"42"}`)

	w := createRequest(t, s.ChatHandler, api.ChatRequest{
		Model:    "think-format-stream",
		Messages: []api.Message{{Role: "user", Content: "Answer in JSON."}},
		Think:    &api.ThinkValue{Value: true},
		Format:   thinkFormatSchema,
		Stream:   &streaming,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	checkOneFormattedCall(t, requests(), []string{"</think>"})

	var thinking, content strings.Builder
	lastThinking, firstContent := -1, -1
	var final api.ChatResponse
	scanner := bufio.NewScanner(w.Body)
	for i := 0; scanner.Scan(); i++ {
		var chunk api.ChatResponse
		if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
			t.Fatalf("chunk %d: %v", i, err)
		}
		if chunk.Message.Thinking != "" {
			thinking.WriteString(chunk.Message.Thinking)
			lastThinking = i
		}
		if strings.TrimSpace(chunk.Message.Content) != "" {
			content.WriteString(chunk.Message.Content)
			if firstContent < 0 {
				firstContent = i
			}
		}
		if !chunk.Done && (chunk.Metrics.PromptEvalCount != 0 || chunk.Metrics.EvalCount != 0) {
			t.Errorf("chunk %d carries metrics before the end: %+v", i, chunk.Metrics)
		}
		if chunk.Done {
			final = chunk
		}
	}
	if lastThinking < 0 || firstContent < 0 || lastThinking >= firstContent {
		t.Errorf("thinking chunks end at %d and content starts at %d, want all thinking first", lastThinking, firstContent)
	}
	if got := thinking.String(); got != "Let me think." {
		t.Errorf("streamed thinking = %q, want %q", got, "Let me think.")
	}
	if got := strings.TrimSpace(content.String()); got != `{"answer":"42"}` {
		t.Errorf("streamed content = %q, want the JSON", got)
	}
	if !final.Done {
		t.Fatal("no final chunk")
	}
	checkPassedThroughMetrics(t, final.Metrics)
}

// TestGenerateThinkFormatLength checks that a response whose thinking runs
// out of num_predict before it closes ends length-limited with no response,
// from the one call and with the runner's metrics.
func TestGenerateThinkFormatLength(t *testing.T) {
	gin.SetMode(gin.TestMode)
	stream := false
	mock := &mockRunner{}
	s := setupImplicitThinkingModel(t, mock, "think-format-length", "nemotron-3-nano")
	requests := recordCompletions(mock, llm.DoneReasonLength, "Endless reasoning")

	w := createRequest(t, s.GenerateHandler, api.GenerateRequest{
		Model:  "think-format-length",
		Prompt: "Answer in JSON.",
		Think:  &api.ThinkValue{Value: true},
		Format: thinkFormatSchema,
		Stream: &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	checkOneFormattedCall(t, requests(), []string{"</think>"})

	var resp api.GenerateResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.DoneReason != "length" {
		t.Errorf("done_reason = %q, want length", resp.DoneReason)
	}
	if resp.Response != "" {
		t.Errorf("response = %q, want none: the thinking never closed", resp.Response)
	}
	if resp.Thinking != "Endless reasoning" {
		t.Errorf("thinking = %q, want the thinking produced", resp.Thinking)
	}
	checkPassedThroughMetrics(t, resp.Metrics)
}
