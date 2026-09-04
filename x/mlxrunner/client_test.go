package mlxrunner

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
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

func TestClientCompletionRequestsIntermediateMetrics(t *testing.T) {
	var request CompletionRequest
	want := CompletionResponse{
		Done:                  true,
		PromptEvalCount:       10,
		PromptEvalCachedCount: testIntPtr(4),
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		if err := json.NewEncoder(w).Encode(want); err != nil {
			t.Errorf("encode response: %v", err)
		}
	}))
	t.Cleanup(srv.Close)

	_, portString, err := net.SplitHostPort(srv.Listener.Addr().String())
	if err != nil {
		t.Fatalf("parse server port: %v", err)
	}
	port, err := strconv.Atoi(portString)
	if err != nil {
		t.Fatalf("parse server port: %v", err)
	}
	client := &Client{port: port, client: srv.Client()}
	opts := api.DefaultOptions()
	var got llm.CompletionResponse
	if err := client.Completion(t.Context(), llm.CompletionRequest{
		Options:                    &opts,
		IncludeIntermediateMetrics: true,
	}, func(response llm.CompletionResponse) { got = response }); err != nil {
		t.Fatalf("Completion: %v", err)
	}
	if !request.IncludeIntermediateMetrics {
		t.Fatal("metrics per token was not forwarded to the MLX runner")
	}
	if got.PromptEvalCount != want.PromptEvalCount || got.PromptEvalCachedCount == nil || *got.PromptEvalCachedCount != *want.PromptEvalCachedCount {
		t.Errorf("prompt counts = (%d, %v), want (%d, %d)", got.PromptEvalCount, got.PromptEvalCachedCount, want.PromptEvalCount, *want.PromptEvalCachedCount)
	}
}
