package gemma4

import (
	"fmt"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

var (
	_ base.MediaModel       = (*Model)(nil)
	_ base.MediaBudgetModel = (*Model)(nil)
)

// resolveImageBudget applies gemma4's own defaults to an unset budget, using the
// same convention as llm/llama_server.go: a non-positive value means unset. For
// gemma4 the shared api defaults *are* its defaults (ADR 0008's ladder), so the
// resolution is direct here; models with their own ceiling — glimmer's 4096 —
// must substitute theirs instead of inheriting these.
func resolveImageBudget(imageMinTokens, imageMaxTokens int) (minTok, maxTok int) {
	minTok, maxTok = imageMinTokens, imageMaxTokens
	if minTok <= 0 {
		minTok = api.DefaultImageMinTokens
	}
	if maxTok <= 0 {
		maxTok = api.DefaultImageMaxTokens
	}
	return minTok, maxTok
}

// PrepareMedia implements base.MediaModel with gemma4's default budget. The
// runner calls PrepareMediaWithBudget instead whenever a request carries one;
// this exists so gemma4 still satisfies the upstream interface.
func (m *Model) PrepareMedia(segments []base.Segment) (*base.PreparedRequest, error) {
	return m.PrepareMediaWithBudget(segments, 0, 0)
}

// PrepareMediaWithBudget implements base.MediaBudgetModel: splice each image
// segment's expansion — BOI, one placeholder per soft token, EOI — into the
// stream, decoding and patchifying on the CPU at the resolved budget.
//
// The expansion is marked non-causal: gemma4's image spans attend
// bidirectionally, so a chunk must cover a whole expansion in one forward
// rather than resuming inside it.
func (m *Model) PrepareMediaWithBudget(segments []base.Segment, imageMinTokens, imageMaxTokens int) (*base.PreparedRequest, error) {
	minTok, maxTok := resolveImageBudget(imageMinTokens, imageMaxTokens)
	boi, placeholder, eoi := m.VisionTokens()

	prepared := &base.PreparedRequest{}
	for s, seg := range segments {
		if seg.Data == nil {
			prepared.Tokens = append(prepared.Tokens, seg.Tokens...)
			continue
		}
		if !m.SupportsVision() {
			return nil, fmt.Errorf("this model does not support %s input", seg.Kind)
		}
		// Kept at the model rather than the runner: upstream routes the kind
		// through Segment and lets each model refuse what it cannot serve, so
		// this is where gemma4's "images only" guarantee now lives.
		if seg.Kind != "image" {
			return nil, fmt.Errorf("gemma4 does not support %s input", seg.Kind)
		}

		vi, err := m.newVisionInput(seg.Data, minTok, maxTok)
		if err != nil {
			return nil, fmt.Errorf("preprocess image: %w", err)
		}

		start := len(prepared.Tokens)
		prepared.Tokens = append(prepared.Tokens, boi)
		for range vi.soft {
			prepared.Tokens = append(prepared.Tokens, placeholder)
		}
		prepared.Tokens = append(prepared.Tokens, eoi)

		prepared.Items = append(prepared.Items, base.PreparedItem{
			Range:  [2]int{start, len(prepared.Tokens)},
			Source: s,
			// Dims is the encoder input shape, and it reaches cache identity:
			// the budget changes the resize target, which changes the patch
			// count, so two budgets over identical bytes never share a key.
			MediaData: vi.patches,
			Dims:      []int{1, int(vi.n), int(vi.patchDim)},
			Opaque:    vi,
			Causal:    false,
		})
	}
	return prepared, nil
}

// EncodeMedia implements base.MediaModel: run the vision path over one prepared
// image, returning the lazy [1, SoftTokens, hidden] features. Nothing evaluates
// here — the consuming forward pulls the graph.
//
// The pixels come from data rather than the item's Opaque buffer: the runner
// frees MediaData once the expansion is evaluated.
func (m *Model) EncodeMedia(item *base.PreparedItem, data *mlx.Array) *mlx.Array {
	vi := item.Opaque.(*visionInput)
	x := data.AsType(mlx.DTypeBFloat16)
	var h *mlx.Array
	if m.VisionEmbedder != nil {
		h = m.VisionEmbedder.Forward(x, vi.xs, vi.ys)
	} else {
		h = m.VisionTower.Forward(x, vi.xs, vi.ys, vi.gridW, vi.gridH, m.VisionCfg)
	}
	return m.projectVision(h)
}
