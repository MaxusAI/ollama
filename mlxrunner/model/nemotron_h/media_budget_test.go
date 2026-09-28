package nemotron_h

import (
	"bytes"
	"image"
	"image/png"
	"slices"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlxrunner/model"
)

// omniVisionConfig is Nemotron 3 Omni's c-radio_v4-h preprocessing: 16px patches,
// a 2×2 shuffle, 1024…13312 patches (256…3328 tokens).
func omniVisionConfig() *VisionConfig {
	return &VisionConfig{
		PatchSize:        16,
		DownsampleFactor: 2,
		MinNumPatches:    1024,
		MaxNumPatches:    13312,
		MaxModelLen:      nemotronVisionDefaultMaxModelLen,
		Std:              [3]float32{1, 1, 1},
	}
}

func budgetTestModel(cfg *VisionConfig) *Model {
	return &Model{
		VisionEncoder:     &RadioVisionEncoder{},
		Projector:         &VisionProjector{},
		VisionConfig:      cfg,
		imageStartTokenID: 10,
		imageTokenID:      11,
		imageEndTokenID:   12,
	}
}

func pngOfSize(t *testing.T, width, height int) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, image.NewNRGBA(image.Rect(0, 0, width, height))); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// prepareOne prepares a single image and returns its item and feature-token count
// (the expansion minus the image start and end markers).
func prepareOne(t *testing.T, m *Model, data []byte, minTok, maxTok int) (model.PreparedItem, int) {
	t.Helper()
	prepared, err := m.PrepareMediaWithBudget([]model.Segment{{Kind: "image", Data: data}}, minTok, maxTok)
	if err != nil {
		t.Fatal(err)
	}
	if len(prepared.Items) != 1 {
		t.Fatalf("items = %d, want 1", len(prepared.Items))
	}
	item := prepared.Items[0]
	return item, item.Range[1] - item.Range[0] - 2
}

func TestImagePatchBounds(t *testing.T) {
	for _, tt := range []struct {
		name             string
		cfg              *VisionConfig
		minTok, maxTok   int
		wantMin, wantMax int
	}{
		{name: "unset keeps the model's own", minTok: 0, maxTok: 0, wantMin: 1024, wantMax: 13312},
		{name: "the shared api default counts as unset", minTok: api.DefaultImageMinTokens, maxTok: api.DefaultImageMaxTokens, wantMin: 1024, wantMax: 13312},
		{name: "max maps tokens to patches x4", maxTok: 1024, wantMin: 1024, wantMax: 4096},
		{name: "max below the model floor is honoured", maxTok: 128, wantMin: 512, wantMax: 512},
		{name: "min raises the floor", minTok: 1024, wantMin: 4096, wantMax: 13312},
		{name: "max clamps to the trained ceiling", maxTok: 100000, wantMin: 1024, wantMax: 13312},
		{name: "min clamps down to max", minTok: 2000, maxTok: 1000, wantMin: 4000, wantMax: 4000},
		{
			name:    "the context-bound ceiling still binds",
			cfg:     &VisionConfig{DownsampleFactor: 2, MinNumPatches: 1024, MaxNumPatches: 13312, MaxModelLen: 512},
			maxTok:  3000,
			wantMin: 1024, wantMax: 2032,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cfg := tt.cfg
			if cfg == nil {
				cfg = omniVisionConfig()
			}
			gotMin, gotMax := budgetTestModel(cfg).imagePatchBounds(tt.minTok, tt.maxTok)
			if gotMin != tt.wantMin || gotMax != tt.wantMax {
				t.Fatalf("imagePatchBounds(%d, %d) = (%d, %d), want (%d, %d)",
					tt.minTok, tt.maxTok, gotMin, gotMax, tt.wantMin, tt.wantMax)
			}
		})
	}
}

// An unset request must reproduce upstream's preprocessing exactly: the grid
// upstream's own PrepareMedia computed before the budget existed.
func TestPrepareMediaWithBudgetUnsetMatchesUpstream(t *testing.T) {
	cfg := omniVisionConfig()
	m := budgetTestModel(cfg)
	for _, size := range [][2]int{{320, 240}, {1920, 1080}, {2048, 2048}, {512, 2048}} {
		w, h := size[0], size[1]
		data := pngOfSize(t, w, h)

		gh, gw := nemotronImagePatchGrid(h, w, nemotronImagePatchBudget(cfg), cfg)
		want := []int{1, 3, gh * int(cfg.PatchSize), gw * int(cfg.PatchSize)}

		unset, unsetTokens := prepareOne(t, m, data, 0, 0)
		shared, sharedTokens := prepareOne(t, m, data, api.DefaultImageMinTokens, api.DefaultImageMaxTokens)
		if !slices.Equal(unset.Dims, want) || !slices.Equal(shared.Dims, want) {
			t.Fatalf("%dx%d: dims unset=%v shared-default=%v, want upstream's %v", w, h, unset.Dims, shared.Dims, want)
		}
		if unsetTokens != sharedTokens || unsetTokens != m.visionTokenCount(want[2], want[3]) {
			t.Fatalf("%dx%d: tokens unset=%d shared-default=%d, want %d", w, h, unsetTokens, sharedTokens, m.visionTokenCount(want[2], want[3]))
		}
		if !slices.Equal(unset.MediaData, shared.MediaData) {
			t.Fatalf("%dx%d: pixels differ between an unset request and the shared default", w, h)
		}

		viaMediaModel, err := m.PrepareMedia([]model.Segment{{Kind: "image", Data: data}})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(viaMediaModel.Items[0].Dims, want) {
			t.Fatalf("%dx%d: PrepareMedia dims %v, want %v", w, h, viaMediaModel.Items[0].Dims, want)
		}
	}
}

func TestPrepareMediaWithBudgetHonoursTheRequest(t *testing.T) {
	cfg := omniVisionConfig()
	m := budgetTestModel(cfg)

	large := pngOfSize(t, 2048, 2048)
	def, defTokens := prepareOne(t, m, large, 0, 0)
	if defTokens > 3328 {
		t.Fatalf("default: %d tokens, above the model's 3328 ceiling", defTokens)
	}

	capped, cappedTokens := prepareOne(t, m, large, 0, 1024)
	if cappedTokens > 1024 || cappedTokens < 900 {
		t.Fatalf("image_max_tokens=1024: %d tokens, want (900, 1024]", cappedTokens)
	}
	// The budget reaches cache identity through the geometry: a different grid means
	// different dims and a different expansion length, so the two never share a prefix.
	if slices.Equal(capped.Dims, def.Dims) || cappedTokens == defTokens {
		t.Fatalf("image_max_tokens=1024 left the geometry unchanged: dims %v tokens %d", capped.Dims, cappedTokens)
	}

	if _, tiny := prepareOne(t, m, large, 0, 128); tiny > 128 {
		t.Fatalf("image_max_tokens=128: %d tokens, want <= 128 (honoured below the model's floor, as on GGUF)", tiny)
	}

	if _, over := prepareOne(t, m, large, 0, 100000); over != defTokens {
		t.Fatalf("image_max_tokens above the ceiling: %d tokens, want the default's %d", over, defTokens)
	}

	small := pngOfSize(t, 320, 240)
	_, smallDefault := prepareOne(t, m, small, 0, 0)
	raised, raisedTokens := prepareOne(t, m, small, 1024, 0)
	gh, gw := nemotronImagePatchGrid(240, 320, nemotronImagePatchBudget(cfg), withMinNumPatches(cfg, 4096))
	if want := []int{1, 3, gh * 16, gw * 16}; !slices.Equal(raised.Dims, want) {
		t.Fatalf("image_min_tokens=1024: dims %v, want %v", raised.Dims, want)
	}
	if raisedTokens < 1000 || raisedTokens <= smallDefault {
		t.Fatalf("image_min_tokens=1024: %d tokens (default %d), want an upscale to about 1024", raisedTokens, smallDefault)
	}
}

func TestPrepareMediaWithBudgetReportsDisabledVision(t *testing.T) {
	m := &Model{}
	if _, err := m.PrepareMediaWithBudget([]model.Segment{{Kind: "image", Data: []byte{1}}}, 256, 1024); err == nil {
		t.Fatal("a model without vision accepted an image with a budget")
	}
	prepared, err := m.PrepareMediaWithBudget([]model.Segment{{Tokens: []int32{1, 2}}}, 256, 1024)
	if err != nil {
		t.Fatalf("text-only request with a budget: %v", err)
	}
	if !slices.Equal(prepared.Tokens, []int32{1, 2}) {
		t.Fatalf("tokens = %v, want [1 2]", prepared.Tokens)
	}
}
