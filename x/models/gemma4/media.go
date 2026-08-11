package gemma4

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/mlxrunner/batch"
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

// softRun returns the absolute [start, end) positions of an item's placeholder
// run. PreparedItem.Range brackets the expansion with BOI and EOI, which carry
// no features, so the feature rows line up with Range shifted one to the right
// and shortened by two — getting this offset wrong shifts every image feature by
// a token and degrades output without failing anything.
func softRun(item batch.MediaItem) (start, end int) {
	vi := item.Opaque.(*visionInput)
	return item.Pos + 1, item.Pos + 1 + vi.soft
}

// bidiSpans returns the absolute [start, end) ranges that attend
// bidirectionally: the placeholder run of every non-causal media item this batch
// carries. It replaces batch.BidiSpans, which the upstream media rework left
// without a writer.
func bidiSpans(b *batch.Batch) [][2]int32 {
	var spans [][2]int32
	for _, item := range b.Media {
		if _, ok := item.Opaque.(*visionInput); !ok {
			continue
		}
		start, end := softRun(item)
		spans = append(spans, [2]int32{int32(start), int32(end)})
	}
	return spans
}

// visionMaskData builds the dense [L, K] additive mask for a chunk starting at
// absolute position off: causal, window-limited, and fully connected within each
// bidirectional span.
//
// Positions are absolute on both axes (q = off + qi), so a span is honoured
// wherever the chunk resumes — the property that lets an image block ride a
// mid-prompt chunk. Split out from visionChunkMask so it can be tested without
// building a batch or touching MLX.
func visionMaskData(spans [][2]int32, off, L, K, window int) []float32 {
	inSpan := func(p int32) int {
		for si, s := range spans {
			if p >= s[0] && p < s[1] {
				return si
			}
		}
		return -1
	}

	neg := float32(math.Inf(-1))
	data := make([]float32, L*K)
	for qi := range L {
		q := off + qi
		qs := inSpan(int32(q))
		for k := range K {
			allowed := k <= q && (window <= 0 || q-k < window)
			if !allowed && qs >= 0 && inSpan(int32(k)) == qs {
				allowed = true
			}
			if !allowed {
				data[qi*K+k] = neg
			}
		}
	}
	return data
}

// scatterMedia overwrites the placeholder rows this chunk covers with the
// image's features, replacing the pre-merge MergedEmbeddings path. Features are
// spliced unscaled, matching the reference's get_input_embeddings: only the text
// embeddings carry EmbedScale.
//
// A chunk may cover part of an expansion, so both sides are clipped to the
// overlap of the item's run with this forward's query range.
func (m *Model) scatterMedia(h *mlx.Array, b *batch.Batch) *mlx.Array {
	for _, item := range b.Media {
		if item.Features == nil {
			continue
		}
		start, end := softRun(item)
		off := int(b.SeqOffsets[item.Seq])
		qLo := max(start, off)
		qHi := min(end, off+int(b.SeqQueryLens[item.Seq]))
		if qHi <= qLo {
			continue
		}

		feat := item.Features.Slice(mlx.Slice(), mlx.Slice(qLo-start, qHi-start), mlx.Slice())
		h = h.SliceUpdate(feat.AsType(h.DType()),
			mlx.Slice(item.Seq, item.Seq+1), mlx.Slice(qLo-off, qHi-off), mlx.Slice())
	}
	return h
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
