package tools

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// enableCloudForTest makes the cloud policy hermetic.
//
// WebFetch.Execute is gated on internalcloud.Disabled() -> envconfig.NoCloud(),
// which reads OLLAMA_NO_CLOUD *and* ~/.ollama/server.json. Without this, these
// tests inherit whatever the host machine happens to have configured: on a box
// where cloud is disabled they fail with "ollama cloud is disabled: web fetch
// is unavailable" before reaching the behaviour under test. That is how this
// surfaced on the windows-latest runner while passing on macOS and Ubuntu.
//
// USERPROFILE matters as much as HOME — os.UserHomeDir() reads it on Windows,
// so setting only HOME leaves the real profile in play there. Mirrors
// internal/cloud's setTestHome.
func enableCloudForTest(t *testing.T) {
	t.Helper()
	home := t.TempDir() // no .ollama/server.json, so nothing disables cloud
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("OLLAMA_NO_CLOUD", "")
	envconfig.ReloadServerConfig()
	// t.Setenv restores the environment, but the parsed config is cached
	// package-globally — reload again so the temp home does not leak into
	// later tests.
	t.Cleanup(envconfig.ReloadServerConfig)
}

func TestWebToolsRequireApproval(t *testing.T) {
	if !coreagent.ToolRequiresApproval((&WebSearch{}), map[string]any{"query": "ollama"}) {
		t.Fatal("web search should require approval")
	}
	if !coreagent.ToolRequiresApproval((&WebFetch{}), map[string]any{"url": "https://ollama.com"}) {
		t.Fatal("web fetch should require approval")
	}
}

func TestWebFetchRejectsUnsupportedScheme(t *testing.T) {
	// Without this the cloud gate short-circuits Execute and every case returns
	// the same "cloud is disabled" error — which still satisfies wantErr, so the
	// scheme assertions would pass while testing nothing.
	enableCloudForTest(t)

	tests := []struct {
		name    string
		url     string
		wantErr bool
	}{
		{name: "file scheme", url: "file:///etc/passwd", wantErr: true},
		{name: "data scheme", url: "data:text/plain,secret", wantErr: true},
		{name: "ftp scheme", url: "ftp://example.com/secret", wantErr: true},
		{name: "http allowed", url: "http://example.com", wantErr: false},
		{name: "https allowed", url: "https://example.com", wantErr: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := (&WebFetch{}).Execute(t.Context(), coreagent.ToolContext{}, map[string]any{"url": tt.url})
			if tt.wantErr && err == nil {
				t.Fatal("expected unsupported scheme to be rejected")
			}
			// For allowed schemes we expect an error only from the missing
			// server/auth path, not from scheme validation. The http/https
			// cases reach the client and may fail on connection/auth; we only
			// assert that the error is NOT a scheme error.
			if !tt.wantErr && err != nil && strings.Contains(err.Error(), "unsupported URL scheme") {
				t.Fatalf("http/https rejected as unsupported: %v", err)
			}
		})
	}
}

func TestWebFetchBoundsContentBeforeReturning(t *testing.T) {
	enableCloudForTest(t)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/experimental/web_fetch" {
			t.Fatalf("path = %q, want /api/experimental/web_fetch", r.URL.Path)
		}
		var req api.WebFetchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatal(err)
		}
		if req.URL != "https://ollama.com" {
			t.Fatalf("request URL = %q, want https://ollama.com", req.URL)
		}
		if err := json.NewEncoder(w).Encode(api.WebFetchResponse{
			Title:   "Ollama",
			Content: strings.Repeat("x", maxWebFetchContentRunes+25),
		}); err != nil {
			t.Fatal(err)
		}
	}))
	defer ts.Close()
	t.Setenv("OLLAMA_HOST", ts.URL)

	result, err := (&WebFetch{}).Execute(t.Context(), coreagent.ToolContext{}, map[string]any{
		"url": "https://ollama.com",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "[tool output truncated: showing first ~") ||
		!strings.Contains(result.Content, "omitted ~7 tokens") ||
		!strings.Contains(result.Content, "Use a narrower request or search query") {
		t.Fatalf("content missing truncation marker: %q", result.Content)
	}
	if count := strings.Count(result.Content, "x"); count != maxWebFetchContentRunes {
		t.Fatalf("captured content count = %d, want %d", count, maxWebFetchContentRunes)
	}
}
