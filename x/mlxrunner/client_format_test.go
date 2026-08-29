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

func TestParseGrammarNeverSilentlyDropsAConstraint(t *testing.T) {
	// ADR 0009's guarantee, re-asserted against upstream's grammar engine
	// after v0.33.2 replaced the fork's compileFormat/Constraint layer: a
	// format the runner cannot honour must be an ERROR, never a silently
	// dropped constraint. The raw-GBNF rejection this file used to assert is
	// now structural -- upstream deleted CompletionRequest.Grammar in
	// 7027546c, so a caller can no longer express one.

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
		spec, err := parseGrammar(c.format)
		if err != nil {
			t.Errorf("parseGrammar(%s): %v", c.name, err)
		}
		if spec != nil {
			t.Errorf("parseGrammar(%s): unexpected constraint", c.name)
		}
	}

	spec, err := parseGrammar(json.RawMessage(`"json"`))
	if err != nil {
		t.Fatalf("parseGrammar(json): %v", err)
	}
	if spec == nil {
		t.Fatal("parseGrammar(json): no constraint")
	}

	if _, err := parseGrammar(json.RawMessage(`"yaml"`)); err == nil {
		t.Fatal("parseGrammar(yaml): expected an error, not a dropped constraint")
	}
	if _, err := parseGrammar(json.RawMessage(`{"type":`)); err == nil {
		t.Fatal("parseGrammar(malformed schema): expected an error")
	}
}
