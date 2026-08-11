package base

import (
	"testing"

	"github.com/ollama/ollama/api"
)

// TestResolveImageBudgetKeepsTheModelsOwnCeiling is the guard for phase 1b's
// whole point. api.DefaultOptions always populates the budget fields with
// gemma4's ladder, so every request carries one whether the caller meant it or
// not. A model with its own ceiling — glimmer's 4096, chosen because lowering it
// hurts OCR — must therefore treat the shared default as "unset", or default
// traffic silently downsizes its images.
func TestResolveImageBudgetKeepsTheModelsOwnCeiling(t *testing.T) {
	const modelMin, modelMax = 64, 4096

	for _, tc := range []struct {
		name             string
		reqMin, reqMax   int
		wantMin, wantMax int
	}{
		{"unset resolves to the model's own", 0, 0, modelMin, modelMax},
		{"negative resolves to the model's own", -1, -1, modelMin, modelMax},
		{
			"shared api default counts as unset",
			api.DefaultImageMinTokens, api.DefaultImageMaxTokens,
			modelMin, modelMax,
		},
		{"an explicitly different value wins", 128, 2048, 128, 2048},
		{"only the ceiling set", 0, 2048, modelMin, 2048},
		{"only the floor set", 128, 0, 128, modelMax},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gotMin, gotMax := ResolveImageBudget(tc.reqMin, tc.reqMax, modelMin, modelMax)
			if gotMin != tc.wantMin || gotMax != tc.wantMax {
				t.Fatalf("ResolveImageBudget(%d, %d, %d, %d) = (%d, %d), want (%d, %d)",
					tc.reqMin, tc.reqMax, modelMin, modelMax, gotMin, gotMax, tc.wantMin, tc.wantMax)
			}
		})
	}
}

// TestResolveImageBudgetCollapsesForTheDefaultsOwner gemma4's own bounds are the
// shared defaults, so resolution is a no-op there: an explicit default and an
// unset budget must both land on the same value, or the two paths would disagree
// for the model the defaults were written for.
func TestResolveImageBudgetCollapsesForTheDefaultsOwner(t *testing.T) {
	unsetMin, unsetMax := ResolveImageBudget(0, 0, api.DefaultImageMinTokens, api.DefaultImageMaxTokens)
	explicitMin, explicitMax := ResolveImageBudget(
		api.DefaultImageMinTokens, api.DefaultImageMaxTokens,
		api.DefaultImageMinTokens, api.DefaultImageMaxTokens)

	if unsetMin != explicitMin || unsetMax != explicitMax {
		t.Fatalf("unset (%d, %d) != explicit default (%d, %d)",
			unsetMin, unsetMax, explicitMin, explicitMax)
	}
	if unsetMax != api.DefaultImageMaxTokens {
		t.Fatalf("resolved ceiling = %d, want the api default %d", unsetMax, api.DefaultImageMaxTokens)
	}
}
