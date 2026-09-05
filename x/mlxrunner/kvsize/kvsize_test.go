package kvsize

import (
	"os"
	"path/filepath"
	"testing"
)

// The four rungs the vision suite's ladder climbs
// (docs/maxusai/spec/vision-harness-reuse.md H4a).
var ladder = []int{8192, 16384, 32768, 65536}

func load(t *testing.T, name string) []byte {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	return data
}

const (
	gemma4Config      = "gemma4-26b-config.json"
	gemma4DraftConfig = "gemma4-26b-draft-config.json"
	qwenMoEConfig     = "qwen3_5-moe-35b-a3b-config.json"
	qwenDenseConfig   = "qwen3_5-27b-config.json"
	nemotronConfig    = "nemotron_h-synthetic-config.json"
	llamaConfig       = "llama-dense-synthetic-config.json"
)

// Per-kind layer counts. Each expectation is derived from the fixture's own
// fields plus the rule the matching model package applies, and the derivation is
// stated so a future reader can re-check it against the config rather than
// against this test.
func TestLayerCountsPerFixture(t *testing.T) {
	cases := []struct {
		name   string
		config string
		draft  string
		want   LayerCounts
		why    string
	}{
		{
			name:   "gemma4 26b",
			config: gemma4Config,
			draft:  gemma4DraftConfig,
			want:   LayerCounts{Attention: 5, Sliding: 25},
			why: "text_config.num_hidden_layers=30 with layer_types holding 25 " +
				"\"sliding_attention\" and 5 \"full_attention\" (every 6th layer); " +
				"num_kv_shared_layers=0 so no layer donates its KV and all 30 own a " +
				"cache; sliding_window=1024>0 so the sliding ones are rotating. The " +
				"draft is Gemma4AssistantForCausalLM, whose NewCaches returns nil.",
		},
		{
			name:   "qwen3.6 35b-a3b (MoE)",
			config: qwenMoEConfig,
			want:   LayerCounts{Attention: 10, Recurrent: 30},
			why: "text_config.num_hidden_layers=40, layer_types holding 30 " +
				"\"linear_attention\" and 10 \"full_attention\" (full_attention_interval=4, " +
				"so every 4th layer). Linear layers take a recurrent cache, full ones a KV cache.",
		},
		{
			name:   "qwen3.8 27b",
			config: qwenDenseConfig,
			want:   LayerCounts{Attention: 16, Recurrent: 48},
			why: "text_config.num_hidden_layers=64, layer_types holding 48 " +
				"\"linear_attention\" and 16 \"full_attention\" (interval 4).",
		},
		{
			name:   "nemotron_h synthetic",
			config: nemotronConfig,
			want:   LayerCounts{Attention: 2, Recurrent: 4},
			why: "hybrid_override_pattern \"M-M*M-MA\": four 'M' recurrent layers, one " +
				"'*' and one 'A' attention layer, and two '-' layers that own no cache.",
		},
		{
			name:   "llama synthetic",
			config: llamaConfig,
			want:   LayerCounts{Attention: 4},
			why:    "num_hidden_layers=4 and no sliding layers: llama takes a full KV cache per layer.",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var draft []byte
			if tc.draft != "" {
				draft = load(t, tc.draft)
			}
			got := Model(load(t, tc.config), draft, 8192)
			if !got.Known {
				t.Fatalf("architecture %q was not priced", got.Arch)
			}
			if got.Layers != tc.want {
				t.Errorf("layer counts = %+v, want %+v\n%s", got.Layers, tc.want, tc.why)
			}
		})
	}
}

// A hand-computed byte total for one rung per fixture. The arithmetic is spelled
// out; if the estimator changes, one of these numbers has to change with a
// reason, not a re-run.
func TestHandComputedTotals(t *testing.T) {
	cases := []struct {
		name                                     string
		config, draft                            string
		numCtx                                   int
		attention, sliding, recurrent, elemBytes uint64
		arithmetic                               string
	}{
		{
			name: "gemma4 26b at 8192", config: gemma4Config, draft: gemma4DraftConfig,
			numCtx: 8192, elemBytes: 2,
			// 5 full layers. attention_k_eq_v=true so a full layer has no
			// v_proj and uses num_global_key_value_heads=2 with
			// global_head_dim=512; K and V are both stored.
			//   2 arrays * 2 heads * 512 dim * 8192 slots * 2 B = 33,554,432 B
			//   * 5 layers                                      = 167,772,160 B
			attention: 167_772_160,
			// 25 sliding layers, bounded by sliding_window=1024 (not 8192),
			// num_key_value_heads=8, head_dim=256.
			//   2 * 8 * 256 * 1024 * 2 = 8,388,608 B * 25 = 209,715,200 B
			sliding: 209_715_200,
			// Draft adds nothing: the assistant owns no cache.
			arithmetic: "167,772,160 + 209,715,200 = 377,487,360 B (360 MiB)",
		},
		{
			name: "qwen3.6 35b-a3b at 8192", config: qwenMoEConfig, numCtx: 8192, elemBytes: 2,
			// 10 full layers: 2 * 2 heads * 256 dim * 8192 * 2 B = 16,777,216 B
			//   * 10 = 167,772,160 B
			attention: 167_772_160,
			// 30 linear layers. conv_dim = 2*16*128 + 32*128 = 8192.
			//   conv  = (4-1) * 8192 * 2 B                = 49,152 B
			//   delta = 32 * 128 * 128 * 4 B (always f32) = 2,097,152 B
			//   (49,152 + 2,097,152) * 30                 = 64,389,120 B
			recurrent:  64_389_120,
			arithmetic: "167,772,160 + 64,389,120 = 232,161,280 B",
		},
		{
			name: "qwen3.8 27b at 8192", config: qwenDenseConfig, numCtx: 8192, elemBytes: 2,
			// 16 full layers: 2 * 4 heads * 256 dim * 8192 * 2 B = 33,554,432 B
			//   * 16 = 536,870,912 B
			attention: 536_870_912,
			// 48 linear layers. conv_dim = 2*16*128 + 48*128 = 10,240.
			//   conv  = 3 * 10,240 * 2 B     = 61,440 B
			//   delta = 48 * 128 * 128 * 4 B = 3,145,728 B
			//   (61,440 + 3,145,728) * 48    = 153,944,064 B
			recurrent:  153_944_064,
			arithmetic: "536,870,912 + 153,944,064 = 690,814,976 B",
		},
		{
			name: "nemotron_h synthetic at 8192", config: nemotronConfig, numCtx: 8192, elemBytes: 2,
			// 2 attention layers: 2 * 2 heads * 128 dim * 8192 * 2 B = 8,388,608 B
			//   * 2 = 16,777,216 B
			attention: 16_777_216,
			// 4 Mamba layers. conv_dim = 16*64 + 2*2*128 = 1,536.
			//   conv  = (4-1) * 1,536 * 2 B  = 9,216 B
			//   delta = 16 * 64 * 128 * 4 B  = 524,288 B
			//   (9,216 + 524,288) * 4        = 2,134,016 B
			recurrent:  2_134_016,
			arithmetic: "16,777,216 + 2,134,016 = 18,911,232 B",
		},
		{
			name: "llama synthetic at 8192", config: llamaConfig, numCtx: 8192, elemBytes: 4,
			// head_dim is absent, so it derives as hidden_size/num_attention_heads
			// = 512/8 = 64, and torch_dtype float32 makes an element 4 bytes.
			//   2 * 2 heads * 64 dim * 8192 * 4 B = 8,388,608 B * 4 layers
			attention:  33_554_432,
			arithmetic: "33,554,432 B",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var draft []byte
			if tc.draft != "" {
				draft = load(t, tc.draft)
			}
			got := Model(load(t, tc.config), draft, tc.numCtx)
			if uint64(got.ElemBytes) != tc.elemBytes {
				t.Errorf("elem bytes = %d, want %d", got.ElemBytes, tc.elemBytes)
			}
			if got.Attention != tc.attention {
				t.Errorf("attention = %d, want %d", got.Attention, tc.attention)
			}
			if got.Sliding != tc.sliding {
				t.Errorf("sliding = %d, want %d", got.Sliding, tc.sliding)
			}
			if got.Recurrent != tc.recurrent {
				t.Errorf("recurrent = %d, want %d", got.Recurrent, tc.recurrent)
			}
			want := tc.attention + tc.sliding + tc.recurrent
			if got.Total() != want {
				t.Errorf("total = %d, want %d (%s)", got.Total(), want, tc.arithmetic)
			}
		})
	}
}

// The point of the whole exercise: the estimate has to move with the rung, or
// admission cannot tell 8192 from 65536.
func TestTotalIsMonotonicInNumCtx(t *testing.T) {
	for _, name := range []string{gemma4Config, qwenMoEConfig, qwenDenseConfig, nemotronConfig, llamaConfig} {
		t.Run(name, func(t *testing.T) {
			config := load(t, name)
			var prev uint64
			for _, numCtx := range ladder {
				got := Model(config, nil, numCtx).Total()
				if got <= prev {
					t.Fatalf("num_ctx %d: total %d did not exceed the previous rung's %d", numCtx, got, prev)
				}
				prev = got
			}
		})
	}
}

// A rotating cache is bounded by its window, so past the window the sliding
// half of the estimate stops growing. Getting this wrong is how a 31b at 65536
// gets refused on a card that would have served it.
func TestSlidingLayersFlattenAtTheWindow(t *testing.T) {
	config := load(t, gemma4Config)

	// sliding_window = 1024, so every rung at or above 1024 holds the same
	// 1024 slots.
	atWindow := Model(config, nil, 1024).Sliding
	for _, numCtx := range ladder {
		if got := Model(config, nil, numCtx).Sliding; got != atWindow {
			t.Errorf("num_ctx %d: sliding = %d, want the windowed %d", numCtx, got, atWindow)
		}
	}

	// Below the window it must still track the rung, or the bound has been
	// applied as a constant rather than as a ceiling.
	if small := Model(config, nil, 512).Sliding; small != atWindow/2 {
		t.Errorf("num_ctx 512: sliding = %d, want half of the windowed %d", small, atWindow)
	}
}

// Recurrent state is per-layer constant: no rung changes it.
func TestRecurrentIsConstantAcrossRungs(t *testing.T) {
	for _, name := range []string{qwenMoEConfig, qwenDenseConfig, nemotronConfig} {
		t.Run(name, func(t *testing.T) {
			config := load(t, name)
			want := Model(config, nil, ladder[0]).Recurrent
			if want == 0 {
				t.Fatal("fixture has no recurrent layers; it cannot prove anything here")
			}
			for _, numCtx := range ladder[1:] {
				if got := Model(config, nil, numCtx).Recurrent; got != want {
					t.Errorf("num_ctx %d: recurrent = %d, want the constant %d", numCtx, got, want)
				}
			}
		})
	}
}

// An architecture with no rule must report Known == false and price nothing.
// The caller falls back to weights-only admission; refusing an unpriced model
// would be strictly worse than today.
func TestUnknownArchitectureIsNotPriced(t *testing.T) {
	config := []byte(`{"architectures":["SomeArchWeHaveNeverSeenForCausalLM"],
	  "model_type":"never_seen","num_hidden_layers":48,"num_key_value_heads":8,"head_dim":128}`)
	got := Model(config, nil, 65536)
	if got.Known {
		t.Fatal("an unregistered architecture must not be priced")
	}
	if got.Total() != 0 {
		t.Errorf("total = %d, want 0", got.Total())
	}
	if got.Arch != "SomeArchWeHaveNeverSeenForCausalLM" {
		t.Errorf("arch = %q, want the name so the warning can carry it", got.Arch)
	}
}

// A config that names a known architecture but does not carry what the rule
// needs is unknown, not zero: nemotron_h without its hybrid pattern cannot be
// priced, and pricing it as all-attention would over-refuse.
func TestKnownArchitectureWithAnUnusableConfigIsUnknown(t *testing.T) {
	config := []byte(`{"architectures":["NemotronHForCausalLM"],"num_hidden_layers":8,
	  "num_attention_heads":8,"hidden_size":1024}`)
	if got := Model(config, nil, 8192); got.Known {
		t.Fatalf("a nemotron_h config with no hybrid_override_pattern must not be priced, got %+v", got)
	}
}

// Dispatch falls back to model_type when architectures is absent, exactly as
// base.New does.
func TestModelTypeFallback(t *testing.T) {
	config := []byte(`{"model_type":"gemma4_unified","text_config":{
	  "num_hidden_layers":2,"num_key_value_heads":1,"head_dim":128,"layer_types":["full_attention","full_attention"]}}`)
	got := Model(config, nil, 256)
	if !got.Known || got.Layers.Attention != 2 {
		t.Fatalf("model_type dispatch failed: %+v", got)
	}
}

// The gemma4 draft is priced as owning no cache, and that is the code's
// decision, not an omission: AssistantModel.NewCaches returns nil.
func TestGemma4DraftOwnsNoCache(t *testing.T) {
	target := load(t, gemma4Config)
	draft := load(t, gemma4DraftConfig)

	with := Model(target, draft, 32768)
	without := Model(target, nil, 32768)

	if !with.DraftKnown {
		t.Errorf("draft arch %q was not recognised", with.DraftArch)
	}
	if with.Total() != without.Total() {
		t.Errorf("draft added %d bytes; the assistant keeps no KV", with.Total()-without.Total())
	}
}

// gemma4's KV sharing decides how many layers own a cache at all, and the two
// fixtures sit on either side of the branch: the 26b shares nothing, its draft
// declares num_kv_shared_layers == num_hidden_layers.
func TestGemma4KVShareTruncation(t *testing.T) {
	target, err := parse(load(t, gemma4Config))
	if err != nil {
		t.Fatal(err)
	}
	if got := gemma4CacheLayers(target); got != 30 {
		t.Errorf("26b cache layers = %d, want all 30 (num_kv_shared_layers=0)", got)
	}

	// The draft declares num_kv_shared_layers=4 over 4 layers, so the first
	// shared layer is index 0 and there are no earlier layers to donate: the
	// donor search finds nothing and every layer keeps its own cache
	// (gemma4.go:501-520). This is the edge the truncation must not
	// mis-handle by assuming "the last N layers are free".
	draft, err := parse(load(t, gemma4DraftConfig))
	if err != nil {
		t.Fatal(err)
	}
	if got := gemma4CacheLayers(draft); got != 4 {
		t.Errorf("draft cache layers = %d, want 4: no donors exist before layer 0", got)
	}

	// A tail that does find donors truncates: 6 layers, the last 2 shared,
	// each matching an earlier layer of its own type, so NewCaches stops at 4.
	shared := []byte(`{"architectures":["Gemma4ForCausalLM"],"text_config":{
	  "num_hidden_layers":6,"num_kv_shared_layers":2,"num_key_value_heads":4,
	  "head_dim":128,"sliding_window":512,
	  "layer_types":["sliding_attention","sliding_attention","full_attention",
	                 "sliding_attention","sliding_attention","full_attention"]}}`)
	got := Model(shared, nil, 8192)
	if want := (LayerCounts{Attention: 1, Sliding: 3}); got.Layers != want {
		t.Errorf("layers = %+v, want %+v: the two shared tail layers own no cache", got.Layers, want)
	}
}

// Every full attention cache rounds its buffer up to a whole 256-token block,
// because that is how cache.KVCache.appendKV grows.
func TestSlotsRoundUpToTheCacheStep(t *testing.T) {
	for _, tc := range []struct{ numCtx, want int }{
		{0, 0}, {1, 256}, {255, 256}, {256, 256}, {257, 512}, {8192, 8192}, {8193, 8448},
	} {
		if got := slots(tc.numCtx); got != tc.want {
			t.Errorf("slots(%d) = %d, want %d", tc.numCtx, got, tc.want)
		}
	}
}
