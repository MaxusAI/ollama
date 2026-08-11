package mlxrunner

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/fnv"
	"log/slog"
	"regexp"
	"strconv"

	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

var imgTagPattern = regexp.MustCompile(`\[img-(\d+)\]`)

// mediaItem is one media occurrence in a request's token stream: the
// absolute position and length of its placeholder expansion, its trie-key
// fold value, and the prepared item.
type mediaItem struct {
	pos    int
	length int
	fold   uint32
	item   *base.PreparedItem
}

// foldValue derives the trie-key substitute for a media item: a hash of the
// raw bytes and the geometry words the caller folds in — the preprocessing
// dims and the expansion length, which together pin the features for given
// bytes — with bit 31 forced so it can never equal a token ID.
func foldValue(data []byte, dims []int) uint32 {
	h := fnv.New64a()
	h.Write(data)
	var b [8]byte
	for _, d := range dims {
		binary.LittleEndian.PutUint64(b[:], uint64(d))
		h.Write(b[:])
	}
	sum := h.Sum64()
	return (uint32(sum>>32) ^ uint32(sum)) | 1<<31
}

// requestMedia manages one request's media features: encoded on first
// use, released when the expansion is fully evaluated. A nil
// *requestMedia is a text-only request; every method is nil-safe.
type requestMedia struct {
	model    base.MediaModel
	items    []mediaItem
	inputLen int

	// manifest is the request-scoped batch view of items; Features is
	// toggled in place so every batch shares the same slice.
	manifest []batch.MediaItem
	features []*mlx.Array // parallel to items; nil until encoded

	// layout is the request's one-row Batch.Layout, shared by every batch
	// like the manifest; nil when the model returned no layout.
	layout []any
}

func (r *Runner) openMedia(request Request) *requestMedia {
	if len(request.MediaItems) == 0 {
		return nil
	}
	m := &requestMedia{
		model:    r.Model.(base.MediaModel),
		items:    request.MediaItems,
		inputLen: len(request.Tokens),
		manifest: make([]batch.MediaItem, len(request.MediaItems)),
		features: make([]*mlx.Array, len(request.MediaItems)),
	}
	if request.Layout != nil {
		m.layout = []any{request.Layout}
	}
	for i, item := range m.items {
		m.manifest[i] = batch.MediaItem{Pos: item.pos, Opaque: item.item.Opaque}
	}
	return m
}

// rowLayout returns the request's per-row Batch.Layout value.
func (m *requestMedia) rowLayout() []any {
	if m == nil {
		return nil
	}
	return m.layout
}

func (item *mediaItem) atomic() bool { return !item.item.Causal }

// extendChunk keeps a chunk from ending strictly inside an atomic
// expansion: cut before one starting inside the chunk, else grow to its
// end, clipped one short of the prompt to preserve the decode seed.
func (m *requestMedia) extendChunk(pos, n int) int {
	if m == nil {
		return n
	}

	end := pos + n
	for i := range m.items {
		item := &m.items[i]
		if !item.atomic() {
			continue
		}
		if item.pos < end && end < item.pos+item.length {
			if item.pos > pos {
				return item.pos - pos
			}
			return min(item.pos+item.length, m.inputLen-1) - pos
		}
	}
	return n
}

// growOpeningChunk grows chunk zero to cover every bidirectional expansion
// (ADR 0014). Fork-local, and deliberately separate from extendChunk so
// upstream's rule keeps its exact semantics and its tests.
//
// A model serving a non-causal expansion attends over the chunk's own k/v
// rather than the cache history — gemma4 does, because routing through history
// lets the sliding applier re-add the window over relaxed image blocks. That is
// only sound while the chunk holds the whole key set, i.e. while it starts at
// position zero. extendChunk keeps a chunk from *ending* inside an expansion but
// happily resumes inside one, which would leave a block attending over a partial
// prefix: not an error, just silently wrong attention, and downstream it
// surfaced as garbage logits rather than anything nameable.
//
// Causal media needs none of this and is not charged for it.
func (m *requestMedia) growOpeningChunk(pos, n int) int {
	if m == nil || pos != 0 {
		return n
	}
	last := 0
	for i := range m.items {
		if item := &m.items[i]; item.atomic() {
			last = max(last, item.pos+item.length)
		}
	}
	if last == 0 {
		return n
	}
	// Clipped one short of the prompt so decode keeps its seed token.
	return max(n, min(last, m.inputLen-1)-pos)
}

// batchMedia returns the manifest for chunk [pos, pos+n), encoding and
// pinning each item's features on first overlap; nothing evaluates here —
// the consuming forward pulls the encoder.
func (m *requestMedia) batchMedia(pos, n int) []batch.MediaItem {
	if m == nil {
		return nil
	}
	for i, item := range m.items {
		if item.pos >= pos+n || item.pos+item.length <= pos {
			continue
		}
		if m.features[i] == nil {
			data := mlx.FromValues(item.item.MediaData, item.item.Dims...)
			m.features[i] = m.model.EncodeMedia(item.item, data)
			mlx.Pin(m.features[i])
			// The upload copied the pixels; free them here — release never
			// passes the end of an expansion reaching the prompt's last token.
			item.item.MediaData = nil
		}
		m.manifest[i].Features = m.features[i]
	}
	return m.manifest
}

// release frees what items fully evaluated or restored at position pos no
// longer need: the pinned features and the preprocessed pixel buffer.
func (m *requestMedia) release(pos int) {
	if m == nil {
		return
	}
	for i, item := range m.items {
		if item.pos+item.length <= pos {
			item.item.MediaData = nil
			if m.features[i] != nil {
				mlx.Unpin(m.features[i])
				m.features[i] = nil
				m.manifest[i].Features = nil
			}
		}
	}
}

// close unpins whatever remains when the pipeline exits.
func (m *requestMedia) close() {
	if m == nil {
		return
	}
	for i, f := range m.features {
		if f != nil {
			mlx.Unpin(f)
			m.features[i] = nil
			m.manifest[i].Features = nil
		}
	}
}

// visionPrefillMaskBudget bounds the dense attention mask a bidirectional media
// request can force a model to allocate (ADR 0014).
//
// A non-causal expansion cannot be served by a plain causal mask, so the model
// materializes a dense [chunkLen, keyCount] float32 overlay. keyCount is the
// cache length, which grows with the prompt, so the cost is chunk × context —
// not chunk². At the 2 KiB prefill chunk that is 0.25 GiB at a 32k prompt and
// 1 GiB at 128k, and extendChunk can grow the chunk to cover a whole expansion,
// doubling it again. Without an admission check a single long prompt with a
// trailing image is an unauthenticated allocation of several GiB.
//
// This replaces the pre-merge check that keyed on request.VisionSpans. Phase 3
// removed the position-zero chunk requirement it was written alongside, but the
// dense overlay it guarded is unchanged, so the ceiling still earns its keep.
const visionPrefillMaskBudget = 1 << 30 // 1 GiB

// checkVisionPrefillBudget refuses a request whose bidirectional expansions
// would force a dense mask past the budget. Causal media (glimmer, qwen3.5)
// needs no overlay and is not charged.
func checkVisionPrefillBudget(items []mediaItem, promptLen int) error {
	chunk := 0
	for _, it := range items {
		if it.item.Causal {
			continue
		}
		// The opening chunk grows to the end of the last bidirectional
		// expansion (extendChunk), so the widest mask is that far in — not
		// merely one expansion long.
		chunk = max(chunk, max(prefillChunkSize(), it.pos+it.length))
	}
	if chunk == 0 {
		return nil
	}

	const bytesPerEntry = 4 // float32 mask entries
	if bytes := int64(chunk) * int64(promptLen) * bytesPerEntry; bytes > visionPrefillMaskBudget {
		return fmt.Errorf(
			"image request needs a %.1f GiB attention mask (%d-token prompt); "+
				"reduce the prompt length or the image-token budget",
			float64(bytes)/(1<<30), promptLen)
	}
	return nil
}

// expandMedia tokenizes the [img-N]-tagged prompt into segments, expands
// them in a single PrepareMedia call, and validates the authored items
// before keying cache identity on them.
func (r *Runner) expandMedia(mm base.MediaModel, prompt string, media []llm.MediaData, imageMinTokens, imageMaxTokens int) (*base.PreparedRequest, []mediaItem, error) {
	matches := imgTagPattern.FindAllStringSubmatch(prompt, -1)
	parts := imgTagPattern.Split(prompt, -1)

	referenced := make([]bool, len(media))
	var segments []base.Segment
	for i, part := range parts {
		segments = append(segments, base.Segment{Tokens: r.Tokenizer.Encode(part, i == 0 && r.Tokenizer.AddBOS())})
		if i >= len(matches) {
			continue
		}

		id, _ := strconv.Atoi(matches[i][1])
		idx := -1
		for j := range media {
			if media[j].ID == id {
				idx = j
				break
			}
		}
		if idx < 0 {
			return nil, nil, fmt.Errorf("invalid image index: %d", id)
		}
		referenced[idx] = true
		segments = append(segments, base.Segment{Kind: string(media[idx].Kind), Data: media[idx].Data})
	}

	for j := range media {
		if !referenced[j] {
			slog.Warn("media not referenced by prompt", "id", media[j].ID)
		}
	}

	// A model that honours the image-token budget gets it; one that does not
	// would silently drop it, which ADR 0009 forbids. Every MLX media model
	// implements the budget interface, so the fallback is for upstream models
	// added between merges rather than a supported state.
	var prepared *base.PreparedRequest
	var err error
	if bm, ok := mm.(base.MediaBudgetModel); ok {
		prepared, err = bm.PrepareMediaWithBudget(segments, imageMinTokens, imageMaxTokens)
	} else {
		slog.Warn("model does not honour the image-token budget; using its own defaults",
			"image_min_tokens", imageMinTokens, "image_max_tokens", imageMaxTokens)
		prepared, err = mm.PrepareMedia(segments)
	}
	if err != nil {
		return nil, nil, err
	}
	items, err := bindItems(prepared, segments)
	if err != nil {
		return nil, nil, err
	}
	return prepared, items, nil
}

// bindItems validates the authored ranges before cache identity is keyed
// on them and binds each item to its source segment's bytes.
func bindItems(prepared *base.PreparedRequest, segments []base.Segment) ([]mediaItem, error) {
	covered := make([]bool, len(segments))
	items := make([]mediaItem, 0, len(prepared.Items))
	end := 0
	for i := range prepared.Items {
		item := &prepared.Items[i]
		rg := item.Range
		if rg[0] < end || rg[1] <= rg[0] || rg[1] > len(prepared.Tokens) {
			return nil, fmt.Errorf("media expansion has invalid range %v", rg)
		}
		if item.Source < 0 || item.Source >= len(segments) || segments[item.Source].Data == nil {
			return nil, fmt.Errorf("media expansion references non-media segment %d", item.Source)
		}
		covered[item.Source] = true
		end = rg[1]

		// The expansion length is the resolved image-token budget for these
		// bytes (ADR 0003/0007/0008), so it belongs in cache identity beside
		// the dims: a budget that moves the soft-token count without moving
		// the preprocessing dims would otherwise fold identically at both
		// budgets, making the smaller one's key sequence a strict prefix of
		// the larger one's and letting a restore splice KV computed from
		// different embeddings.
		length := rg[1] - rg[0]
		geometry := make([]int, 0, len(item.Dims)+1)
		geometry = append(geometry, item.Dims...)
		geometry = append(geometry, length)

		items = append(items, mediaItem{
			pos:    rg[0],
			length: length,
			fold:   foldValue(segments[item.Source].Data, geometry),
			item:   item,
		})
	}
	for s, seg := range segments {
		if seg.Data != nil && !covered[s] {
			return nil, errors.New("media expansion produced no tokens")
		}
	}
	return items, nil
}
