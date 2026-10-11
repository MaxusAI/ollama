package server

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/types/model"
)

func TestModelOptionsNumCtxPriority(t *testing.T) {
	tests := []struct {
		name           string
		envContextLen  string // empty means not set (uses 0 sentinel)
		defaultNumCtx  int    // VRAM-based default
		modelNumCtx    int    // 0 means not set in model
		requestNumCtx  int    // 0 means not set in request
		expectedNumCtx int
	}{
		{
			name:           "vram default when nothing else set",
			envContextLen:  "",
			defaultNumCtx:  32768,
			modelNumCtx:    0,
			requestNumCtx:  0,
			expectedNumCtx: 32768,
		},
		{
			name:           "env var overrides vram default",
			envContextLen:  "8192",
			defaultNumCtx:  32768,
			modelNumCtx:    0,
			requestNumCtx:  0,
			expectedNumCtx: 8192,
		},
		{
			name:           "model overrides vram default",
			envContextLen:  "",
			defaultNumCtx:  32768,
			modelNumCtx:    16384,
			requestNumCtx:  0,
			expectedNumCtx: 16384,
		},
		{
			name:           "model overrides env var",
			envContextLen:  "8192",
			defaultNumCtx:  32768,
			modelNumCtx:    16384,
			requestNumCtx:  0,
			expectedNumCtx: 16384,
		},
		{
			name:           "request overrides everything",
			envContextLen:  "8192",
			defaultNumCtx:  32768,
			modelNumCtx:    16384,
			requestNumCtx:  4096,
			expectedNumCtx: 4096,
		},
		{
			name:           "request overrides vram default",
			envContextLen:  "",
			defaultNumCtx:  32768,
			modelNumCtx:    0,
			requestNumCtx:  4096,
			expectedNumCtx: 4096,
		},
		{
			name:           "request overrides model",
			envContextLen:  "",
			defaultNumCtx:  32768,
			modelNumCtx:    16384,
			requestNumCtx:  4096,
			expectedNumCtx: 4096,
		},
		{
			name:           "low vram tier default",
			envContextLen:  "",
			defaultNumCtx:  4096,
			modelNumCtx:    0,
			requestNumCtx:  0,
			expectedNumCtx: 4096,
		},
		{
			name:           "high vram tier default",
			envContextLen:  "",
			defaultNumCtx:  262144,
			modelNumCtx:    0,
			requestNumCtx:  0,
			expectedNumCtx: 262144,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set or clear environment variable
			if tt.envContextLen != "" {
				t.Setenv("OLLAMA_CONTEXT_LENGTH", tt.envContextLen)
			}

			// Create server with VRAM-based default
			s := &Server{
				defaultNumCtx: tt.defaultNumCtx,
			}

			// Create model options (use float64 as FromMap expects JSON-style numbers)
			var modelOpts map[string]any
			if tt.modelNumCtx != 0 {
				modelOpts = map[string]any{"num_ctx": float64(tt.modelNumCtx)}
			}
			model := &Model{
				Options: modelOpts,
			}

			// Create request options (use float64 as FromMap expects JSON-style numbers)
			var requestOpts map[string]any
			if tt.requestNumCtx != 0 {
				requestOpts = map[string]any{"num_ctx": float64(tt.requestNumCtx)}
			}

			opts, err := s.modelOptions(model, requestOpts)
			if err != nil {
				t.Fatalf("modelOptions failed: %v", err)
			}

			if opts.NumCtx != tt.expectedNumCtx {
				t.Errorf("NumCtx = %d, want %d", opts.NumCtx, tt.expectedNumCtx)
			}
		})
	}
}

func TestModelOptionsGenerationDefaultsPriority(t *testing.T) {
	m := &Model{
		GenerationDefaults: model.GenerationDefaults{
			"top_k":          int64(12),
			"top_p":          float64(0.7),
			"min_p":          float64(0.05),
			"temperature":    float64(0.4),
			"repeat_last_n":  int64(128),
			"repeat_penalty": float64(1.2),
		},
		Options: map[string]any{
			"top_p":         float64(0.5),
			"min_p":         float64(0),
			"repeat_last_n": float64(0),
		},
	}
	requestOpts := map[string]any{
		"temperature":    float64(0),
		"repeat_penalty": float64(1.5),
	}

	opts, err := (&Server{}).modelOptions(m, requestOpts)
	if err != nil {
		t.Fatal(err)
	}

	if opts.TopK != 12 {
		t.Fatalf("TopK = %d, want 12", opts.TopK)
	}
	if opts.TopP != 0.5 {
		t.Fatalf("TopP = %v, want 0.5", opts.TopP)
	}
	if opts.MinP != 0 {
		t.Fatalf("MinP = %v, want 0", opts.MinP)
	}
	if opts.Temperature != 0 {
		t.Fatalf("Temperature = %v, want 0", opts.Temperature)
	}
	if opts.RepeatLastN != 0 {
		t.Fatalf("RepeatLastN = %d, want 0", opts.RepeatLastN)
	}
	if opts.RepeatPenalty != 1.5 {
		t.Fatalf("RepeatPenalty = %v, want 1.5", opts.RepeatPenalty)
	}
}

func TestModelOptionsEmbeddingNumBatchDefault(t *testing.T) {
	tests := []struct {
		name             string
		defaultNumCtx    int
		capabilities     []string
		modelOpts        map[string]any
		requestOpts      map[string]any
		expectedNumBatch int
	}{
		{
			name:             "embedding model defaults to embedding batch size",
			defaultNumCtx:    40960,
			capabilities:     []string{string(model.CapabilityEmbedding)},
			expectedNumBatch: llm.DefaultEmbeddingNumBatch,
		},
		{
			name:             "embedding default is capped by context",
			defaultNumCtx:    1024,
			capabilities:     []string{string(model.CapabilityEmbedding)},
			expectedNumBatch: 1024,
		},
		{
			name:             "model num_batch overrides embedding default",
			defaultNumCtx:    40960,
			capabilities:     []string{string(model.CapabilityEmbedding)},
			modelOpts:        map[string]any{"num_batch": float64(1024)},
			expectedNumBatch: 1024,
		},
		{
			name:             "request num_batch overrides embedding default",
			defaultNumCtx:    40960,
			capabilities:     []string{string(model.CapabilityEmbedding)},
			requestOpts:      map[string]any{"num_batch": float64(4096)},
			expectedNumBatch: 4096,
		},
		{
			name:             "non embedding model keeps general default",
			defaultNumCtx:    40960,
			capabilities:     []string{string(model.CapabilityCompletion)},
			expectedNumBatch: 512,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &Server{defaultNumCtx: tt.defaultNumCtx}
			m := &Model{
				Options: tt.modelOpts,
			}
			m.Config.Capabilities = tt.capabilities

			opts, err := s.modelOptions(m, tt.requestOpts)
			if err != nil {
				t.Fatalf("modelOptions failed: %v", err)
			}

			if opts.NumBatch != tt.expectedNumBatch {
				t.Fatalf("NumBatch = %d, want %d", opts.NumBatch, tt.expectedNumBatch)
			}
		})
	}
}

func TestModelOptionsDraftNumPredictDefault(t *testing.T) {
	tests := []struct {
		name        string
		model       *Model
		requestOpts map[string]any
		want        int
	}{
		{
			name:  "separate draft model keeps default enabled",
			model: &Model{DraftPath: "draft.gguf"},
			want:  4,
		},
		{
			name:  "embedded draft requires explicit parameter",
			model: &Model{},
			want:  0,
		},
		{
			name:  "model parameter enables embedded draft",
			model: &Model{Options: map[string]any{"draft_num_predict": float64(4)}},
			want:  4,
		},
		{
			name:        "request parameter enables embedded draft",
			model:       &Model{},
			requestOpts: map[string]any{"draft_num_predict": float64(8)},
			want:        8,
		},
		{
			name:        "request can disable separate draft model",
			model:       &Model{DraftPath: "draft.gguf"},
			requestOpts: map[string]any{"draft_num_predict": float64(0)},
			want:        0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts, err := (&Server{}).modelOptions(tt.model, tt.requestOpts)
			if err != nil {
				t.Fatal(err)
			}
			if opts.DraftNumPredict != tt.want {
				t.Fatalf("DraftNumPredict = %d, want %d", opts.DraftNumPredict, tt.want)
			}
		})
	}
}

func TestUsesAutomaticNumBatch(t *testing.T) {
	tests := []struct {
		name        string
		modelOpts   map[string]any
		requestOpts map[string]any
		want        bool
	}{
		{
			name: "default is automatic",
			want: true,
		},
		{
			name:        "model num_batch is explicit",
			modelOpts:   map[string]any{"num_batch": float64(1024)},
			requestOpts: nil,
			want:        false,
		},
		{
			name:        "request num_batch is explicit",
			requestOpts: map[string]any{"num_batch": float64(2048)},
			want:        false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := usesAutomaticNumBatch(&Model{Options: tt.modelOpts}, tt.requestOpts); got != tt.want {
				t.Fatalf("usesAutomaticNumBatch = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestModelOptionsImageBudgetUnsetIsZero: api.DefaultOptions carries gemma4's
// 70/1120 ladder, so before this an unset image budget and an explicit 70 or
// 1120 reached the runners as the same values, and the runners whose own
// bounds differ (nemotron_h on both paths, muse-glimmer and qwen3.5 on MLX)
// had to read 70/1120 as "unset". An explicit 1120 was therefore not
// expressible there: muse-glimmer:30b-nvfp4 sized a 2048x2048 image at 4096
// tokens under image_max_tokens=1120, and at 1091 under 1119 or 1121
// (2026-10-11). Unset now reaches the runners as 0, which every resolver
// already reads as "the model's own", and any explicit value is honoured.
func TestModelOptionsImageBudgetUnsetIsZero(t *testing.T) {
	for _, tt := range []struct {
		name             string
		model            *Model
		requestOpts      map[string]any
		wantMin, wantMax int
	}{
		{name: "unset reaches the runner as the model's own", model: &Model{}, wantMin: 0, wantMax: 0},
		{name: "an explicit 1120 is kept", model: &Model{}, requestOpts: map[string]any{"image_max_tokens": float64(1120)}, wantMin: 0, wantMax: 1120},
		{name: "an explicit 70 is kept", model: &Model{}, requestOpts: map[string]any{"image_min_tokens": float64(70)}, wantMin: 70, wantMax: 0},
		{name: "a Modelfile value counts as set", model: &Model{Options: map[string]any{"image_max_tokens": float64(560)}}, wantMin: 0, wantMax: 560},
		{name: "the request overrides the Modelfile", model: &Model{Options: map[string]any{"image_max_tokens": float64(560)}}, requestOpts: map[string]any{"image_max_tokens": float64(1120)}, wantMin: 0, wantMax: 1120},
		{name: "no model: the api defaults stand", model: nil, wantMin: api.DefaultImageMinTokens, wantMax: api.DefaultImageMaxTokens},
	} {
		t.Run(tt.name, func(t *testing.T) {
			opts, err := (&Server{}).modelOptions(tt.model, tt.requestOpts)
			if err != nil {
				t.Fatal(err)
			}
			if opts.ImageMinTokens != tt.wantMin || opts.ImageMaxTokens != tt.wantMax {
				t.Fatalf("image budget = %d/%d, want %d/%d", opts.ImageMinTokens, opts.ImageMaxTokens, tt.wantMin, tt.wantMax)
			}
		})
	}
}
