package model

import (
	// Every model's PrepareMedia decodes through image.Decode; the decoder
	// set is registered once here so all models accept the same formats.
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/webp"

	"github.com/ollama/ollama/mlx"
)

// Segment is one run of the prompt in stream order: either a tokenized text
// run (Tokens set) or a single media item (Kind and Data set).
type Segment struct {
	Tokens []int32
	Kind   string
	Data   []byte
}

// PreparedItem describes one media occurrence in the prepared stream. A
// model chooses item granularity: one per media segment, or several when
// parts encode and evaluate independently (e.g. per tile).
type PreparedItem struct {
	// Range is the expansion's token range [start, end) in Tokens;
	// non-empty, since cache identity enters through these positions.
	Range [2]int

	// Source is the index of the segment this item was prepared from; the
	// item's prefix-cache identity is keyed on that segment's bytes.
	Source int

	// MediaData is the preprocessed encoder input with shape Dims. Dims
	// enters the cache keys too: geometry changes features under
	// identical bytes.
	MediaData []float32
	Dims      []int

	// Opaque carries model-private preprocessing state to EncodeMedia
	// and Forward.
	Opaque any

	// Causal marks an expansion whose tokens attend causally, so chunks
	// may split it. Unset, the first evaluation covers the whole
	// expansion in one forward, as bidirectional runs require.
	Causal bool
}

// PreparedRequest is the expanded input stream, every media segment's
// expansion spliced in place, with the items in stream order.
type PreparedRequest struct {
	Tokens []int32
	Items  []PreparedItem

	// Layout is an opaque request-scoped value computed in the one pass
	// that sees every splice position; immutable, carried unread by the
	// runner to every forward. Delivered only when Items is non-empty.
	// Nil when the model derives nothing from it.
	Layout any
}

// MediaModel is implemented by models that accept media inputs.
type MediaModel interface {
	// PrepareMedia runs once per request on the request goroutine, CPU
	// only, and returns the expanded stream. It must be deterministic for
	// given segments: prefix-cache restores splice cached state with
	// recomputed state.
	PrepareMedia(segments []Segment) (*PreparedRequest, error)

	// EncodeMedia builds one item's lazy feature graph on the MLX thread;
	// it must not evaluate — the consuming forward's evaluation pulls it.
	// Read the pixels from data: the runner frees the item's MediaData
	// once its expansion is evaluated.
	EncodeMedia(item *PreparedItem, data *mlx.Array) *mlx.Array
}

// MediaBudgetModel is a MediaModel that honours the per-request image-token
// budget (api.Options.ImageMinTokens / ImageMaxTokens).
//
// Fork-local. Upstream's PrepareMedia takes no options, so a budget sent with a
// request has nowhere to arrive; adding a second interface rather than changing
// MediaModel keeps upstream's shape intact, so a future merge conflicts on this
// block alone instead of on every media model.
//
// Two contracts widen relative to PrepareMedia:
//
//   - Determinism is per (segments, budget), not per segments alone. The budget
//     must therefore reach cache identity — both PreparedItem.Dims and the
//     expansion length carry it — so two budgets over identical bytes never
//     share a prefix. Restoring KV captured at another budget is a correctness
//     bug, not a cache-efficiency one.
//   - A model keeps its own ceiling as its default. A value equal to the shared
//     default counts as unset, matching how llm/llama_server.go resolves the
//     same options for nemotron and qwen-VL, so handing glimmer gemma4's 1120
//     does not silently discard detail it deliberately keeps.
//
// The budget is named per media kind on purpose: a Segment may be text, an image
// or audio, and only images carry a token-rung budget today. If a second kind
// ever needs one, collapse the pair into a struct rather than growing the list.
type MediaBudgetModel interface {
	MediaModel

	PrepareMediaWithBudget(segments []Segment, imageMinTokens, imageMaxTokens int) (*PreparedRequest, error)
}

// ResolveImageBudget resolves a request's image-token budget against a model's
// own bounds. A request value <= 0 is unset and takes the model's; anything
// else is the caller's and wins. The server sends 0 for a budget neither the
// request nor the Modelfile set (server/routes.go), so gemma4's 70/1120 -- the
// api defaults -- no longer stand in for "unset": before, a model that keeps a
// different ceiling (muse-glimmer's 4096, qwen3.5's, nemotron_h's) could not be
// asked for exactly 1120, which resolved to its own 4096 (measured on
// muse-glimmer:30b-nvfp4, 2026-10-11). Unset still keeps such a model's own
// ceiling, which is what the sentinel protected.
func ResolveImageBudget(reqMin, reqMax, modelMin, modelMax int) (minTok, maxTok int) {
	minTok, maxTok = reqMin, reqMax
	if minTok <= 0 {
		minTok = modelMin
	}
	if maxTok <= 0 {
		maxTok = modelMax
	}
	return minTok, maxTok
}
