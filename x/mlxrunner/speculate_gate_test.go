package mlxrunner

import (
	"testing"

	sample "github.com/ollama/ollama/x/mlxrunner/sample"
)

func TestDraftUnderGrammarFromEnv(t *testing.T) {
	// An unrecognised value must keep upstream's default rather than disabling drafting: a typo that silently
	// halved structured-output throughput would be found later, as a performance mystery.
	for _, tc := range []struct {
		value string
		want  bool
	}{
		{"", true}, {"1", true}, {"true", true}, {"ON", true}, {"yes", true},
		{"0", false}, {"false", false}, {" off ", false}, {"No", false},
		{"maybe", true}, {"2", true},
	} {
		if got := draftUnderGrammarFromEnv(tc.value); got != tc.want {
			t.Errorf("draftUnderGrammarFromEnv(%q) = %v, want %v", tc.value, got, tc.want)
		}
	}
}

func TestDraftingEnabled(t *testing.T) {
	constrained := Request{Grammar: &grammarCompilation{}}
	plain := Request{}
	logprobs := Request{SamplerOpts: sample.Options{Logprobs: true}}
	topLogprobs := Request{SamplerOpts: sample.Options{TopLogprobs: 3}}

	for _, tc := range []struct {
		name    string
		knob    bool
		request Request
		want    bool
	}{
		{"upstream's default drafts under a grammar", true, constrained, true},
		{"the knob restores main's gate", false, constrained, false},
		{"an unconstrained request is untouched by the knob", false, plain, true},
		{"logprobs never draft, knob on", true, logprobs, false},
		{"logprobs never draft, knob off", false, logprobs, false},
		{"top logprobs never draft", true, topLogprobs, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			restore := draftUnderGrammar
			draftUnderGrammar = tc.knob
			defer func() { draftUnderGrammar = restore }()
			if got := draftingEnabled(tc.request); got != tc.want {
				t.Errorf("draftingEnabled() = %v, want %v", got, tc.want)
			}
		})
	}
}
