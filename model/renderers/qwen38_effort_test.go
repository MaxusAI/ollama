package renderers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// TestQwen38ReasoningEffortMapping pins which think values emit which effort
// directive. The mapping is not obvious and has already been misread once:
// ThinkValue.String() reports "medium" for the bool true, and the renderer
// treats "medium" as *no directive at all* rather than a middle setting.
//
// It matters for measurement provenance (ADR 0011/0023): a benchmark labelled
// "model default" must say which of these it actually ran. Over HTTP the
// answer is "no directive", because server/routes.go coerces a nil think to
// true for thinking-capable models — so the nil branch below is reachable only
// from direct Go callers, never from the API.
func TestQwen38ReasoningEffortMapping(t *testing.T) {
	msgs := []api.Message{{Role: "user", Content: "hi"}}

	const (
		none  = "none"
		low   = "low"
		xhigh = "xhigh"
	)

	for _, tc := range []struct {
		name  string
		think *api.ThinkValue
		want  string
	}{
		{"nil is xhigh, but unreachable over HTTP", nil, xhigh},
		{"true (the API default) emits no directive", &api.ThinkValue{Value: true}, none},
		{"medium is indistinguishable from the default", &api.ThinkValue{Value: "medium"}, none},
		{"low", &api.ThinkValue{Value: "low"}, low},
		{"high is the publisher's own default", &api.ThinkValue{Value: "high"}, xhigh},
		{"max matches high", &api.ThinkValue{Value: "max"}, xhigh},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := RenderWithRenderer("qwen3.8", msgs, nil, tc.think)
			if err != nil {
				t.Fatalf("render: %v", err)
			}
			level := none
			switch {
			case strings.Contains(got, qwen38XHighReasoningInstructions):
				level = xhigh
			case strings.Contains(got, qwen38LowReasoningInstructions):
				level = low
			}
			if level != tc.want {
				t.Errorf("effort directive = %s, want %s", level, tc.want)
			}
		})
	}
}
