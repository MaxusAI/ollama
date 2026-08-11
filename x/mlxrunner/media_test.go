package mlxrunner

import (
	"slices"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

func TestEffectiveKeyTokens(t *testing.T) {
	tokens := []int32{10, 20, 500, 500, 500, 30}
	items := []mediaItem{{pos: 2, length: 3, fold: foldValue([]byte("img"), []int{1})}}

	eff := effectiveKeyTokens(tokens, items)
	want := []uint32{10, 20, items[0].fold, items[0].fold, items[0].fold, 30}
	if !slices.Equal(eff, want) {
		t.Fatalf("got %v, want %v", eff, want)
	}

	// Text-only streams never alias a media stream: folds carry bit 31.
	for _, e := range effectiveKeyTokens(tokens, nil) {
		if e&(1<<31) != 0 {
			t.Fatalf("token key %d has bit 31 set", e)
		}
	}
}

func TestExtendChunk(t *testing.T) {
	m := &requestMedia{
		items: []mediaItem{
			{pos: 10, length: 4, item: &base.PreparedItem{}},
			{pos: 40, length: 8, item: &base.PreparedItem{Causal: true}},
			{pos: 96, length: 4, item: &base.PreparedItem{}},
		},
		inputLen: 100,
	}

	cases := []struct{ pos, n, want int }{
		{0, 10, 10}, // ends at the expansion start: not inside
		{0, 12, 10}, // expansion starts inside: cut so it begins the next chunk
		{0, 14, 14}, // ends at the expansion end: not inside
		{10, 2, 4},  // chunk starts at the expansion: extend to its end
		{12, 1, 2},  // resume mid-expansion: extend to its end
		{38, 6, 6},  // causal expansion: ending inside is legal
		{42, 4, 4},  // causal expansion at chunk start: no extension
		{90, 7, 6},  // trailing expansion starts inside: cut before it
		{96, 2, 3},  // trailing expansion at chunk start: clip one short of the prompt
	}
	for _, c := range cases {
		if got := m.extendChunk(c.pos, c.n); got != c.want {
			t.Errorf("extendChunk(%d, %d) = %d, want %d", c.pos, c.n, got, c.want)
		}
	}

	var nilMedia *requestMedia
	if got := nilMedia.extendChunk(0, 12); got != 12 {
		t.Errorf("nil extendChunk = %d, want 12", got)
	}
}

// encodeCountingModel counts EncodeMedia calls and returns a real array so
// the pin/release lifecycle runs against live handles.
type encodeCountingModel struct {
	stubMediaModel
	calls *int
}

func (m encodeCountingModel) EncodeMedia(item *base.PreparedItem, data *mlx.Array) *mlx.Array {
	*m.calls++
	return mlx.Zeros(mlx.DTypeFloat32, item.Range[1]-item.Range[0], 4)
}

func TestBatchMediaLifecycle(t *testing.T) {
	skipIfNoMLX(t)

	calls := 0
	prepared := &base.PreparedItem{
		Range:     [2]int{2, 6},
		MediaData: []float32{1, 2},
		Dims:      []int{2},
		Opaque:    7,
	}
	r := &Runner{Model: encodeCountingModel{calls: &calls}}
	request := Request{
		Tokens:     make([]int32, 8),
		MediaItems: []mediaItem{{pos: 2, length: 4, item: prepared}},
	}

	m := r.openMedia(request)
	if m == nil {
		t.Fatal("openMedia returned nil for a media request")
	}
	if m.manifest[0].Pos != 2 || m.manifest[0].Opaque != 7 {
		t.Fatalf("manifest = %+v", m.manifest[0])
	}

	if items := m.batchMedia(0, 2); items[0].Features != nil || calls != 0 {
		t.Fatal("non-overlapping chunk encoded features")
	}
	if items := m.batchMedia(0, 4); items[0].Features == nil || calls != 1 {
		t.Fatalf("overlap did not encode once (calls=%d)", calls)
	}
	if items := m.batchMedia(4, 2); items[0].Features == nil || calls != 1 {
		t.Fatalf("second overlap re-encoded (calls=%d)", calls)
	}

	m.release(4)
	if m.manifest[0].Features == nil {
		t.Fatal("release dropped features before the expansion was evaluated")
	}
	m.release(6)
	if m.manifest[0].Features != nil {
		t.Fatal("release kept features past the expansion end")
	}
	m.close()

	if r.openMedia(Request{Tokens: make([]int32, 8)}) != nil {
		t.Fatal("openMedia returned non-nil for a text-only request")
	}
}

// Two prompts that differ only in their image diverge at the expansion's
// first key — one position earlier under bigram packing — and prompts with
// the same image share keys through the whole expansion.
func TestKeyFoldDivergence(t *testing.T) {
	prompt := func(fold uint32) []uint32 {
		tokens := []int32{1, 2, 900, 900, 900, 3, 4}
		return effectiveKeyTokens(tokens, []mediaItem{{pos: 2, length: 3, fold: fold}})
	}
	imgA := foldValue([]byte("a"), []int{1})
	imgB := foldValue([]byte("b"), []int{1})
	if imgA != foldValue([]byte("a"), []int{1}) {
		t.Fatal("fold not deterministic")
	}
	if imgA == foldValue([]byte("a"), []int{2}) {
		t.Fatal("different dims produced the same fold under identical bytes")
	}

	for _, lookahead := range []int{0, 1} {
		pc := &prefixCache{draftLookahead: lookahead}
		keysA := pc.key(prompt(imgA))
		keysB := pc.key(prompt(imgB))
		keysA2 := pc.key(prompt(imgA))

		if !slices.Equal(keysA, keysA2) {
			t.Fatalf("lookahead %d: same image produced different keys", lookahead)
		}

		// Bigram packing pulls the divergence one position early: the key
		// before the expansion packs (token, fold). Keys re-converge in value
		// after the expansion (shared trailing text), which is fine — the trie
		// paths forked at the first difference.
		divergeAt, convergeAt := 2-lookahead, 5
		for i := range keysA {
			same := keysA[i] == keysB[i]
			if i < divergeAt && !same {
				t.Fatalf("lookahead %d: keys diverge at %d, before the expansion", lookahead, i)
			}
			if i >= divergeAt && i < convergeAt && same {
				t.Fatalf("lookahead %d: keys agree at %d, inside the expansion", lookahead, i)
			}
		}
	}
}

// The resolved image token budget is part of cache identity (ADR 0003): the
// same image preprocessed at two budgets holds different embeddings, so a
// request that lowered image_max_tokens — which does not reload the runner —
// must not restore KV captured at the higher budget. The budget reaches the
// key through the preprocessing dims, which pin the feature geometry, and
// through the expansion length; this pins both.
func TestFoldValueSeparatesBudgets(t *testing.T) {
	data := []byte("image-a")
	small := foldValue(data, []int{4, 256})
	large := foldValue(data, []int{6, 256})
	if small == large {
		t.Fatal("the same image at two budgets folded to one key")
	}

	// The shorter expansion's key stream must share nothing with the longer
	// one's from the expansion's first position on: no prefix of the
	// low-budget stream may match inside the high-budget encoding.
	keysSmall := effectiveKeyTokens([]int32{1, 500, 500, 500, 500, 2}, []mediaItem{{pos: 1, length: 4, fold: small}})
	keysLarge := effectiveKeyTokens([]int32{1, 500, 500, 500, 500, 500, 500, 2}, []mediaItem{{pos: 1, length: 6, fold: large}})
	for i := 1; i < min(len(keysSmall), len(keysLarge)); i++ {
		if keysSmall[i] == keysLarge[i] {
			t.Fatalf("key %d shared across budgets: %d", i, keysSmall[i])
		}
	}
}

// TestCheckVisionPrefillBudget restores ADR 0014's conformance coverage, which
// was lost with the pre-merge implementation. Phase 3 removed the position-zero
// chunk requirement, but the dense [chunk, keys] overlay the ceiling guarded is
// unchanged, so the guard is still load-bearing.
func TestCheckVisionPrefillBudget(t *testing.T) {
	bidi := func(length int) mediaItem {
		return mediaItem{length: length, item: &base.PreparedItem{Causal: false}}
	}
	causal := func(length int) mediaItem {
		return mediaItem{length: length, item: &base.PreparedItem{Causal: true}}
	}

	// The mask is chunk x prompt x 4 bytes; the chunk is at least the prefill
	// chunk size, so the budget is hit by prompt length alone.
	overLen := visionPrefillMaskBudget/(int64(prefillChunkSize())*4) + 1

	for _, tc := range []struct {
		name      string
		items     []mediaItem
		promptLen int
		wantErr   bool
	}{
		{"no media is never charged", nil, int(overLen), false},
		{"causal media needs no dense overlay", []mediaItem{causal(4096)}, int(overLen), false},
		{"short bidi prompt fits", []mediaItem{bidi(256)}, 8192, false},
		{"bidi prompt past the ceiling is refused", []mediaItem{bidi(256)}, int(overLen), true},
		{"a long expansion widens the chunk and so the mask", []mediaItem{bidi(1 << 16)}, 16384, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := checkVisionPrefillBudget(tc.items, tc.promptLen)
			if tc.wantErr && err == nil {
				t.Fatal("want an admission error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("want admission, got %v", err)
			}
		})
	}
}

// TestVisionPrefillBudgetBoundsTheAllocation measures the bound rather than
// asserting it by eye: whatever the check admits must fit the budget.
func TestVisionPrefillBudgetBoundsTheAllocation(t *testing.T) {
	items := []mediaItem{{length: 4096, item: &base.PreparedItem{Causal: false}}}
	for _, promptLen := range []int{1024, 8192, 32768, 131072, 1 << 20} {
		err := checkVisionPrefillBudget(items, promptLen)
		chunk := max(prefillChunkSize(), 4096)
		bytes := int64(chunk) * int64(promptLen) * 4
		if err == nil && bytes > visionPrefillMaskBudget {
			t.Fatalf("admitted a %d-token prompt needing %.2f GiB", promptLen, float64(bytes)/(1<<30))
		}
		if err != nil && bytes <= visionPrefillMaskBudget {
			t.Fatalf("refused a %d-token prompt needing only %.2f GiB", promptLen, float64(bytes)/(1<<30))
		}
	}
}

// TestExtendChunkKeepsBidiBlocksInTheOpeningChunk guards the precondition the
// gemma4 attention path depends on and that no unit test caught the first time:
// a model serving a non-causal expansion attends over the chunk's own k/v, so
// the block must sit in a chunk starting at position zero or it attends over a
// partial prefix. That failure is silent in the mask and only surfaces as
// garbage logits far downstream — it was found by the benchmark suite, not the
// test suite.
func TestExtendChunkKeepsBidiBlocksInTheOpeningChunk(t *testing.T) {
	bidiItem := func(pos, length int) mediaItem {
		return mediaItem{pos: pos, length: length, item: &base.PreparedItem{Causal: false}}
	}
	causalItem := func(pos, length int) mediaItem {
		return mediaItem{pos: pos, length: length, item: &base.PreparedItem{Causal: true}}
	}

	t.Run("opening chunk grows past every bidi expansion", func(t *testing.T) {
		// Three images whose last expansion ends at 3700, well past the 2 KiB
		// prefill chunk — the shape that crashed before this guard existed.
		m := &requestMedia{
			inputLen: 4000,
			items: []mediaItem{
				bidiItem(10, 260), bidiItem(300, 1200), bidiItem(1600, 2100),
			},
		}
		got := m.growOpeningChunk(0, prefillChunkSize())
		if want := 3700; got != want {
			t.Fatalf("opening chunk = %d, want %d (through the last expansion)", got, want)
		}
	})

	t.Run("later chunks are not grown", func(t *testing.T) {
		m := &requestMedia{inputLen: 4000, items: []mediaItem{bidiItem(10, 260)}}
		if got := m.growOpeningChunk(3000, 512); got != 512 {
			t.Fatalf("chunk at 3000 = %d, want 512 untouched", got)
		}
	})

	t.Run("causal media does not grow the opening chunk", func(t *testing.T) {
		// glimmer and qwen3.5 scatter into cache history and need no such
		// guarantee; growing for them would cost memory for nothing.
		m := &requestMedia{inputLen: 4000, items: []mediaItem{causalItem(10, 3000)}}
		if got := m.growOpeningChunk(0, prefillChunkSize()); got != prefillChunkSize() {
			t.Fatalf("opening chunk = %d, want %d for causal media", got, prefillChunkSize())
		}
	})

	t.Run("growth is clipped to keep the decode seed", func(t *testing.T) {
		m := &requestMedia{inputLen: 500, items: []mediaItem{bidiItem(10, 480)}}
		if got := m.growOpeningChunk(0, 499); got != 499 {
			t.Fatalf("opening chunk = %d, want 499 (inputLen-1)", got)
		}
	})
}
