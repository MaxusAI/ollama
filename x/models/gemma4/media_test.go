package gemma4

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

// textSeg and imgSeg build the two Segment shapes PrepareMedia sees.
func textSeg(tokens ...int32) base.Segment { return base.Segment{Tokens: tokens} }
func imgSeg(data []byte) base.Segment      { return base.Segment{Kind: "image", Data: data} }

// visionTestModel is unifiedTestModel with a non-nil embedder so SupportsVision
// reports true. Preprocessing never dereferences it — PrepareMedia is pure Go —
// but the admission gate checks it deliberately, so a checkpoint that shipped no
// vision path is refused before any pixels are decoded rather than failing later
// inside EncodeMedia.
func visionTestModel() *Model {
	m := unifiedTestModel()
	m.VisionEmbedder = &VisionEmbedder{}
	return m
}

// TestPrepareMediaWithBudgetSplicesExpansion pins the expansion's shape: the
// placeholder run is bracketed by BOI/EOI, Range covers the whole bracket, and
// the item is non-causal so a chunk can never resume inside it.
func TestPrepareMediaWithBudgetSplicesExpansion(t *testing.T) {
	m := visionTestModel()
	png := testImagePNG(t, 480, 336)

	prepared, err := m.PrepareMediaWithBudget(
		[]base.Segment{textSeg(1, 2), imgSeg(png), textSeg(3)}, 0, 70)
	if err != nil {
		t.Fatal(err)
	}
	if len(prepared.Items) != 1 {
		t.Fatalf("got %d items, want 1", len(prepared.Items))
	}

	item := prepared.Items[0]
	boi, placeholder, eoi := m.VisionTokens()
	// 2 leading text tokens, then BOI + 70 placeholders + EOI, then 1 trailing.
	if want := 2 + 1 + 70 + 1 + 1; len(prepared.Tokens) != want {
		t.Fatalf("got %d tokens, want %d", len(prepared.Tokens), want)
	}
	if item.Range != [2]int{2, 74} {
		t.Fatalf("Range = %v, want [2 74]", item.Range)
	}
	if prepared.Tokens[item.Range[0]] != boi || prepared.Tokens[item.Range[1]-1] != eoi {
		t.Fatalf("expansion not bracketed by BOI/EOI: %d ... %d",
			prepared.Tokens[item.Range[0]], prepared.Tokens[item.Range[1]-1])
	}
	for i := item.Range[0] + 1; i < item.Range[1]-1; i++ {
		if prepared.Tokens[i] != placeholder {
			t.Fatalf("token %d = %d, want placeholder %d", i, prepared.Tokens[i], placeholder)
		}
	}
	if item.Causal {
		t.Fatal("gemma4 image spans attend bidirectionally, so the item must be non-causal")
	}
	if item.Source != 1 {
		t.Fatalf("Source = %d, want 1 (the image segment)", item.Source)
	}
}

// TestPrepareMediaWithBudgetSeparatesBudgets is the load-bearing one for cache
// identity: the same image bytes at two budgets must differ in both the
// expansion length and Dims. Dims feeds the prefix-cache fold, so if it ever
// stopped moving with the budget a request could restore KV captured at another
// budget — a correctness bug, not a cache-efficiency one.
func TestPrepareMediaWithBudgetSeparatesBudgets(t *testing.T) {
	m := visionTestModel()
	png := testImagePNG(t, 480, 336)

	low, err := m.PrepareMediaWithBudget([]base.Segment{imgSeg(png)}, 0, 70)
	if err != nil {
		t.Fatal(err)
	}
	high, err := m.PrepareMediaWithBudget([]base.Segment{imgSeg(png)}, 0, 280)
	if err != nil {
		t.Fatal(err)
	}

	lo, hi := low.Items[0], high.Items[0]
	if lo.Range[1]-lo.Range[0] >= hi.Range[1]-hi.Range[0] {
		t.Fatalf("expansion did not grow with the budget: %v vs %v", lo.Range, hi.Range)
	}
	if len(low.Tokens) >= len(high.Tokens) {
		t.Fatalf("token count did not grow with the budget: %d vs %d", len(low.Tokens), len(high.Tokens))
	}
	if equalInts(lo.Dims, hi.Dims) {
		t.Fatalf("Dims identical across budgets (%v): two budgets would share a cache prefix", lo.Dims)
	}
}

// TestPrepareMediaDefaultsToTheModelsOwnBudget proves an unset budget resolves
// gemma4's documented default rather than zero or some caller's value — the
// property phase 1b must preserve per-model when glimmer and qwen3_5 are ported.
func TestPrepareMediaDefaultsToTheModelsOwnBudget(t *testing.T) {
	m := visionTestModel()
	png := testImagePNG(t, 480, 336)

	unset, err := m.PrepareMedia([]base.Segment{imgSeg(png)})
	if err != nil {
		t.Fatal(err)
	}
	explicit, err := m.PrepareMediaWithBudget(
		[]base.Segment{imgSeg(png)}, api.DefaultImageMinTokens, api.DefaultImageMaxTokens)
	if err != nil {
		t.Fatal(err)
	}
	if len(unset.Tokens) != len(explicit.Tokens) {
		t.Fatalf("unset budget gave %d tokens, explicit default gave %d",
			len(unset.Tokens), len(explicit.Tokens))
	}
	if !equalInts(unset.Items[0].Dims, explicit.Items[0].Dims) {
		t.Fatalf("unset Dims %v != default Dims %v", unset.Items[0].Dims, explicit.Items[0].Dims)
	}
}

// TestPrepareMediaRejectsNonImage keeps the fork's runner-level audio guarantee
// alive at the seam upstream moved it to: the model refuses what it cannot serve
// rather than letting the bytes reach image.Decode and fail confusingly.
func TestPrepareMediaRejectsNonImage(t *testing.T) {
	m := visionTestModel()
	_, err := m.PrepareMedia([]base.Segment{{Kind: "audio", Data: []byte("not an image")}})
	if err == nil {
		t.Fatal("audio segment accepted; want an explicit refusal")
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
