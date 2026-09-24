package nemotron_h

import "github.com/ollama/ollama/mlxrunner/model"

// Fork-local: the per-request image-token budget for nemotron_h vision (ADR 0021's
// media seam, ADR 0008's options). Upstream's PrepareMedia takes no options, so the
// budget arrives through model.MediaBudgetModel; everything the fork adds lives in
// this file, and upstream's code changes only where PrepareMedia hands the bounds to
// preprocessImage.

var _ model.MediaBudgetModel = (*Model)(nil)

// imagePatchBounds resolves a request's image-token budget into the patch bounds
// nemotronImagePatchGrid works in, with the semantics the GGUF path gives the same
// architecture (nemotronImageTokenBudget in llm/llama_server.go):
//
//   - An unset bound, or one equal to the shared api default (gemma4's ladder, which
//     every request carries), keeps the model's own: min_num_patches, and the
//     context-bound nemotronImagePatchBudget. An unset request therefore reproduces
//     upstream's preprocessing exactly.
//   - One token is DownsampleFactor² patches (the 2×2 pixel shuffle), so tokens map
//     to patches ×4 for Nemotron 3 Omni.
//   - The maximum is clamped to the model's own ceiling — 13312 patches, 3328 tokens,
//     is the trained maximum and raising it buys nothing.
//   - The minimum is clamped down to the maximum. Neither bound is floored at the
//     model's min_num_patches: a request for fewer tokens than the model's default
//     minimum is honoured, as llama-server honours it.
func (m *Model) imagePatchBounds(imageMinTokens, imageMaxTokens int) (minPatches, maxPatches int) {
	cfg := m.VisionConfig
	perToken := int(cfg.DownsampleFactor) * int(cfg.DownsampleFactor)
	ceiling := nemotronImagePatchBudget(cfg)

	minPatches, maxPatches = cfg.MinNumPatches, ceiling
	// Model bounds of 0 mean "keep the model's own", which stay in patch units here
	// so no rounding through tokens can move the default.
	minTok, maxTok := model.ResolveImageBudget(imageMinTokens, imageMaxTokens, 0, 0)
	if maxTok > 0 {
		maxPatches = min(maxTok*perToken, ceiling)
	}
	if minTok > 0 {
		minPatches = minTok * perToken
	}
	return min(minPatches, maxPatches), maxPatches
}

// withMinNumPatches returns a copy of cfg whose floor is minPatches, so upstream's
// nemotronImagePatchGrid applies a request's minimum without changing shape.
// VisionConfig holds values only, so the copy shares nothing with the model's.
func withMinNumPatches(cfg *VisionConfig, minPatches int) *VisionConfig {
	c := *cfg
	c.MinNumPatches = minPatches
	return &c
}
