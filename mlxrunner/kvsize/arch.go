package kvsize

import "strings"

// targetRules maps a target architecture to its cache rule. The keys are the
// names the model packages register with base.Register (x/models/*/init), plus
// the model_type aliases they also register, because dispatch falls back to
// model_type exactly as base.New does.
var targetRules = map[string]rule{
	// x/models/gemma4
	"Gemma4ForCausalLM":                     gemma4Rule,
	"Gemma4ForConditionalGeneration":        gemma4Rule,
	"Gemma4UnifiedForCausalLM":              gemma4Rule,
	"Gemma4UnifiedForConditionalGeneration": gemma4Rule,
	"gemma4_unified":                        gemma4Rule,

	// x/models/qwen3_5 and x/models/qwen3_5_moe (same Model, same NewCaches)
	"Qwen3_5ForCausalLM":                   qwen35Rule,
	"Qwen3_5ForConditionalGeneration":      qwen35Rule,
	"Qwen3NextForCausalLM":                 qwen35Rule,
	"Qwen3NextForConditionalGeneration":    qwen35Rule,
	"Qwen3_5MoeForCausalLM":                qwen35Rule,
	"Qwen3_5MoeForConditionalGeneration":   qwen35Rule,
	"Qwen3NextMoeForCausalLM":              qwen35Rule,
	"Qwen3NextMoeForConditionalGeneration": qwen35Rule,

	// x/models/qwen4_exp
	"Qwen4ExpForConditionalGeneration": qwen4ExpRule,

	// x/models/nemotron_h
	"NemotronHForCausalLM":             nemotronHRule,
	"NemotronH_Nano_VL_V2":             nemotronHRule,
	"NemotronH_Nano_Omni_Reasoning_V3": nemotronHRule,

	// x/models/cohere2_moe, glimmer, laguna: sliding/global splits
	"Cohere2MoeForCausalLM":               cohere2MoeRule,
	"MuseGlimmerForConditionalGeneration": glimmerRule,
	"LagunaForCausalLM":                   lagunaRule,

	// x/models/llama, qwen3: a full KV cache on every layer
	"LlamaForCausalLM": denseRule,
	"Qwen3ForCausalLM": denseRule,

	// x/models/glm4_moe_lite: MLA, one compressed latent per layer
	"Glm4MoeLiteForCausalLM": mlaRule,
	"GLM4MoeLite":            mlaRule,
}

// draftRules maps a draft architecture (base.RegisterDraft) to its rule.
var draftRules = map[string]rule{
	// x/models/gemma4/assistant.go NewCaches returns nil: the assistant keeps
	// no KV of its own and re-attends the target's caches read-only.
	"Gemma4AssistantForCausalLM":        noCacheRule,
	"Gemma4UnifiedAssistantForCausalLM": noCacheRule,
	"gemma4_assistant":                  noCacheRule,
	"gemma4_unified_assistant":          noCacheRule,

	// x/models/dflash: a per-layer context cache, sliding or full.
	"DFlashDraftModel":          dflashRule,
	"DFlashLagunaForCausalLM":   dflashRule,
	"MuseGlimmerAssistantModel": dflashRule,
}

// noCacheRule prices a draft that owns no cache at all.
func noCacheRule(_ *config, _ int) (Estimate, bool) { return Estimate{}, true }

// kvHeads and headDim apply the defaults every model package applies before it
// reshapes K/V: num_key_value_heads falls back to num_attention_heads, head_dim
// to hidden_size / num_attention_heads.
func (c *config) kvHeads() int {
	if c.text.NumKeyValueHeads > 0 {
		return c.text.NumKeyValueHeads
	}
	return c.text.NumAttentionHeads
}

func (c *config) headDim() int {
	if c.text.HeadDim > 0 {
		return c.text.HeadDim
	}
	if c.text.NumAttentionHeads > 0 {
		return c.text.HiddenSize / c.text.NumAttentionHeads
	}
	return 0
}

// gemma4Rule mirrors x/models/gemma4/gemma4.go NewCaches (1116-1134).
//
// Three decisions, all read from the config the same way the model reads them:
//
//   - which layers own a cache at all: NewCaches stops at the first layer with
//     KVShareDonor >= 0, and the donor map is derived from num_kv_shared_layers
//     and layer_types (gemma4.go:499-520). A shared layer whose type has no
//     earlier donor keeps its own cache, which is why this mirrors the search
//     instead of assuming the last num_kv_shared_layers layers are free.
//   - sliding vs global: isLayerSliding (gemma4.go:552-560).
//   - the K/V geometry, which differs per layer type: Attention.Forward
//     (gemma4.go:1228-1287) uses head_dim for sliding layers and
//     global_head_dim for full ones, and a K=V full layer (attention_k_eq_v,
//     no v_proj tensor) uses num_global_key_value_heads instead of
//     num_key_value_heads. Both K and V are still stored -- v is the
//     v-normalised copy of k, a second array.
func gemma4Rule(cfg *config, numCtx int) (Estimate, bool) {
	layers := cfg.text.NumHiddenLayers
	if layers <= 0 {
		return Estimate{}, false
	}

	// Defaults from gemma4.go:420-441.
	headDim := cfg.text.HeadDim
	if headDim == 0 {
		headDim = 256
	}
	globalHeadDim := cfg.text.GlobalHeadDim
	if globalHeadDim == 0 {
		globalHeadDim = headDim
	}
	kvHeads := cfg.text.NumKeyValueHeads
	if kvHeads == 0 {
		kvHeads = 1
	}
	globalKVHeads := kvHeads
	// attention_k_eq_v means the checkpoint ships no v_proj for full layers, so
	// Attention.Forward takes the K=V branch and its global head count.
	if cfg.text.AttentionKEqV && cfg.text.NumGlobalKeyValueHeads > 0 {
		globalKVHeads = cfg.text.NumGlobalKeyValueHeads
	}

	elem := cfg.elemBytes()
	window := cfg.text.SlidingWindow
	cacheLayers := gemma4CacheLayers(cfg)

	var est Estimate
	for i := range cacheLayers {
		if window > 0 && gemma4IsSliding(cfg, i) {
			est.Layers.Sliding++
			est.Sliding += symmetricKV(kvHeads, windowSlots(numCtx, window), headDim, elem)
			continue
		}
		est.Layers.Attention++
		est.Attention += symmetricKV(globalKVHeads, slots(numCtx), globalHeadDim, elem)
	}
	return est, true
}

// gemma4IsSliding mirrors gemma4.go isLayerSliding (552-560), including its
// looser layer_types check (any non-empty list, indexed by layer) and the
// pattern default of 5 applied when neither layer_types nor
// sliding_window_pattern is set (gemma4.go:439-441).
func gemma4IsSliding(cfg *config, i int) bool {
	if len(cfg.text.LayerTypes) > 0 && i < len(cfg.text.LayerTypes) {
		return cfg.text.LayerTypes[i] == "sliding_attention"
	}
	pattern := cfg.text.SlidingWindowPattern
	if pattern <= 0 && len(cfg.text.LayerTypes) == 0 {
		pattern = 5
	}
	if pattern <= 0 {
		return false
	}
	return (i+1)%pattern != 0
}

// gemma4CacheLayers is the number of leading layers that own a cache:
// NewCaches truncates at the first layer with a KV-share donor.
func gemma4CacheLayers(cfg *config) int {
	layers := cfg.text.NumHiddenLayers
	shared := cfg.text.NumKVSharedLayers
	if shared <= 0 || len(cfg.text.LayerTypes) == 0 {
		return layers
	}
	firstShared := layers - shared
	if firstShared < 0 || firstShared > len(cfg.text.LayerTypes) {
		return layers
	}
	prev := cfg.text.LayerTypes[:firstShared]
	for i := firstShared; i < layers && i < len(cfg.text.LayerTypes); i++ {
		for j := len(prev) - 1; j >= 0; j-- {
			if prev[j] == cfg.text.LayerTypes[i] {
				// This layer donates its KV instead of owning one, and
				// NewCaches stops here.
				return i
			}
		}
	}
	return layers
}

// qwen35Rule mirrors x/models/qwen3_5/qwen3_5.go NewCaches (1312-1324): a
// recurrent cache on every linear-attention layer, a full KV cache on the rest.
// Shared by qwen3_5 and qwen3_5_moe, which register the same constructor.
//
// The inline MTP head's own KV cache (mtpDraft.NewCaches, one cache.NewKVCache)
// is deliberately NOT priced: whether the head exists depends on the weights
// having shipped, which the config alone cannot tell us, and one extra
// full-attention layer out of ten to sixteen is well inside the headroom. This
// under-prices rather than over-refuses, which is the bias this package keeps.
func qwen35Rule(cfg *config, numCtx int) (Estimate, bool) {
	layers := cfg.text.NumHiddenLayers
	if layers <= 0 {
		return Estimate{}, false
	}
	elem := cfg.elemBytes()

	var est Estimate
	for i := range layers {
		if qwenIsLinear(cfg, i) {
			est.Layers.Recurrent++
			est.Recurrent += qwenRecurrentBytes(cfg, elem)
			continue
		}
		est.Layers.Attention++
		est.Attention += symmetricKV(cfg.kvHeads(), slots(numCtx), cfg.headDim(), elem)
	}
	return est, true
}

// qwenIsLinear mirrors qwen3_5.go layerIsLinear (356-365) together with the
// full_attention_interval defaults its parse applies (294-307): layer_types
// decides when it covers every layer, otherwise every interval-th layer is a
// full-attention layer and the interval defaults to 4.
func qwenIsLinear(cfg *config, i int) bool {
	if t := cfg.layerType(i); t != "" {
		return !strings.Contains(t, "full")
	}
	interval := cfg.text.FullAttentionInterval
	if interval <= 0 {
		interval = 4
	}
	if interval > cfg.text.NumHiddenLayers {
		interval = cfg.text.NumHiddenLayers
	}
	if interval <= 0 {
		return true
	}
	return (i+1)%interval != 0
}

// qwenRecurrentBytes mirrors the arguments qwen3_5.go NewCaches (1314-1318)
// and qwen4_exp.go NewCaches (96-100) pass to cache.NewRecurrentCache.
func qwenRecurrentBytes(cfg *config, elem int) uint64 {
	convKernel := cfg.text.LinearConvKernelDim
	if convKernel <= 0 {
		convKernel = 4 // qwen3_5.go:253-255
	}
	convDim := 2*cfg.text.LinearNumKeyHeads*cfg.text.LinearKeyHeadDim +
		cfg.text.LinearNumValueHeads*cfg.text.LinearValueHeadDim
	return recurrentBytes(convKernel-1, convDim,
		cfg.text.LinearNumValueHeads, cfg.text.LinearValueHeadDim, cfg.text.LinearKeyHeadDim, elem)
}

// qwen4ExpRule mirrors x/models/qwen4_exp/qwen4_exp.go NewCaches (88-118).
// It is qwen3.5's split plus two things qwen3.5 has not got:
//
//   - one extra full KV cache per full-attention layer, holding the raw QSA
//     indexer keys [1, indexer_kv_heads, T, indexer_head_dim] with the exact
//     3-axis positions as its values. blocks.go:110 writes them; the positions
//     are int32 (qsa.go canonicalRopePositionRows), and KVCache takes the head
//     count from the keys, so the value buffer is [1, indexer_kv_heads, T, 3].
//   - one engram cache: [1, ngram_size-1] int64 token history plus a
//     [1, (ple_conv_kernel_size-1)*ngram_size, hc_count*hidden_size] conv
//     history, both constant in num_ctx (engram_cache.go:51-68).
func qwen4ExpRule(cfg *config, numCtx int) (Estimate, bool) {
	est, ok := qwen35Rule(cfg, numCtx)
	if !ok {
		return est, false
	}

	elem := cfg.elemBytes()
	fullLayers := est.Layers.Attention
	if cfg.text.IndexerKVHeads > 0 && cfg.text.IndexerHeadDim > 0 {
		side := attnBytes(cfg.text.IndexerKVHeads, slots(numCtx), cfg.text.IndexerHeadDim, elem, 3, 4)
		est.Attention += uint64(fullLayers) * side
		est.Layers.Attention += fullLayers
	}

	history := uint64(max(cfg.text.NGramSize-1, 0)) * 8
	convTail := max(cfg.text.PLEConvKernelSize-1, 0) * cfg.text.NGramSize
	convDim := cfg.text.HCCount * cfg.text.HiddenSize
	engram := history + uint64(max(convTail, 0))*uint64(max(convDim, 0))*uint64(elem)
	if engram > 0 {
		est.Recurrent += engram
		est.Layers.Recurrent++
	}
	return est, true
}

// nemotronHRule mirrors x/models/nemotron_h/nemotron_h.go newLayerCache
// (1454-1466): hybrid_override_pattern names each layer's kind. 'M' is a Mamba
// layer with a constant recurrent state, '*' and 'A' are attention layers with
// a full KV cache, and 'E'/'-' (MoE and dense MLP layers) own no cache.
func nemotronHRule(cfg *config, numCtx int) (Estimate, bool) {
	layers := cfg.text.NumHiddenLayers
	pattern := strings.TrimSpace(cfg.text.HybridOverridePattern)
	// parseConfig refuses a pattern that does not cover every layer
	// (nemotron_h.go:245-251), so a mismatch here means we are not looking at a
	// config this rule understands.
	if layers <= 0 || len(pattern) != layers {
		return Estimate{}, false
	}

	// Defaults from nemotron_h.go:196-231.
	convKernel := cfg.text.ConvKernel
	if convKernel <= 0 {
		convKernel = 4
	}
	stateSize := cfg.text.SSMStateSize
	if stateSize <= 0 {
		stateSize = 128
	}
	mambaHeads := cfg.text.MambaNumHeads
	if mambaHeads <= 0 {
		mambaHeads = 128
	}
	mambaHeadDim := cfg.text.MambaHeadDim
	if mambaHeadDim <= 0 {
		mambaHeadDim = 64
	}
	groups := cfg.text.NGroups
	if groups <= 0 {
		groups = 1
	}
	// cfgConvDim (nemotron_h.go:1492-1495).
	convDim := mambaHeads*mambaHeadDim + 2*groups*stateSize

	elem := cfg.elemBytes()
	var est Estimate
	for _, kind := range []byte(pattern) {
		switch kind {
		case 'M':
			est.Layers.Recurrent++
			est.Recurrent += recurrentBytes(convKernel-1, convDim, mambaHeads, mambaHeadDim, stateSize, elem)
		case '*', 'A':
			est.Layers.Attention++
			est.Attention += symmetricKV(cfg.kvHeads(), slots(numCtx), cfg.headDim(), elem)
		}
	}
	return est, true
}

// cohere2MoeRule mirrors x/models/cohere2_moe/cohere2_moe.go NewCaches
// (764-775) and layerIsSliding (298-300), including the layer_types the parse
// derives when the config omits them (254-266): the first first_k_dense_replace
// layers follow prefix_dense_sliding_window_pattern, the rest follow
// sliding_window_pattern, and every pattern-th layer is full attention.
func cohere2MoeRule(cfg *config, numCtx int) (Estimate, bool) {
	layers := cfg.text.NumHiddenLayers
	if layers <= 0 {
		return Estimate{}, false
	}

	// Defaults from cohere2_moe.go:213-221. sliding_window defaults to 4096
	// when the key is absent, which is what makes an omitted window sliding
	// rather than global.
	window := cfg.text.SlidingWindow
	if window <= 0 {
		window = 4096
	}
	slidingPattern := cfg.text.SlidingWindowPattern
	if slidingPattern <= 0 {
		slidingPattern = 4
	}
	prefixPattern := cfg.text.PrefixDenseSlidingWindowPattern
	if prefixPattern <= 0 {
		prefixPattern = 1
	}

	isSliding := func(i int) bool {
		if t := cfg.layerType(i); t != "" {
			return t == "sliding_attention"
		}
		pattern, idx := slidingPattern, i-cfg.text.FirstKDenseReplace
		if i < cfg.text.FirstKDenseReplace {
			pattern, idx = prefixPattern, i
		}
		// patternLayerType (cohere2_moe.go:291-296).
		return !(pattern > 0 && (idx+1)%pattern == 0)
	}
	return slidingSplit(cfg, numCtx, window, isSliding), true
}

// glimmerRule mirrors x/models/glimmer/glimmer.go NewCaches (625-637): the
// layer_types list decides, and the parse rejects a config whose list does not
// cover every layer, so a missing list means every layer is global here.
func glimmerRule(cfg *config, numCtx int) (Estimate, bool) {
	if cfg.text.NumHiddenLayers <= 0 {
		return Estimate{}, false
	}
	isSliding := func(i int) bool { return cfg.layerType(i) == "sliding_attention" }
	return slidingSplit(cfg, numCtx, cfg.text.SlidingWindow, isSliding), true
}

// lagunaRule mirrors x/models/laguna/laguna.go NewCaches (1446-1455) and
// layerIsSliding (521-526), which requires layer_types to cover every layer and
// reports global otherwise.
func lagunaRule(cfg *config, numCtx int) (Estimate, bool) {
	if cfg.text.NumHiddenLayers <= 0 {
		return Estimate{}, false
	}
	isSliding := func(i int) bool { return cfg.layerType(i) == "sliding_attention" }
	return slidingSplit(cfg, numCtx, cfg.text.SlidingWindow, isSliding), true
}

// dflashRule mirrors x/models/dflash/dflash.go NewCaches (532-542): the draft
// keeps its own per-layer context caches, sliding where layer_types says so.
// dflash.go:212-217 defaults an absent layer_types to all-full.
func dflashRule(cfg *config, numCtx int) (Estimate, bool) {
	if cfg.text.NumHiddenLayers <= 0 {
		return Estimate{}, false
	}
	isSliding := func(i int) bool { return cfg.layerType(i) == "sliding_attention" }
	return slidingSplit(cfg, numCtx, cfg.text.SlidingWindow, isSliding), true
}

// denseRule mirrors x/models/llama and x/models/qwen3 NewCaches: one full KV
// cache per layer, num_key_value_heads x head_dim wide.
func denseRule(cfg *config, numCtx int) (Estimate, bool) {
	layers := cfg.text.NumHiddenLayers
	if layers <= 0 {
		return Estimate{}, false
	}
	return slidingSplit(cfg, numCtx, 0, func(int) bool { return false }), true
}

// mlaRule mirrors x/models/glm4_moe_lite/glm4_moe_lite.go: a full KV cache per
// layer (NewCaches 771-778), but MLA stores one compressed latent per token
// rather than per-head K and V. MLAAttention.Forward (118-128) writes keys
// [1, 1, T, kv_lora_rank + qk_rope_head_dim] with a zero-width value array, so
// num_key_value_heads x head_dim would over-price it by an order of magnitude.
func mlaRule(cfg *config, numCtx int) (Estimate, bool) {
	layers := cfg.text.NumHiddenLayers
	latent := cfg.text.KVLoraRank + cfg.text.QKRopeHeadDim
	if layers <= 0 || cfg.text.KVLoraRank <= 0 {
		return Estimate{}, false
	}
	elem := cfg.elemBytes()
	return Estimate{
		Attention: uint64(layers) * attnBytes(1, slots(numCtx), latent, elem, 0, elem),
		Layers:    LayerCounts{Attention: layers},
	}, true
}

// slidingSplit prices the common shape: every layer owns an attention cache,
// rotating when isSliding says so and the model has a window, full otherwise.
func slidingSplit(cfg *config, numCtx, window int, isSliding func(int) bool) Estimate {
	heads, dim, elem := cfg.kvHeads(), cfg.headDim(), cfg.elemBytes()
	var est Estimate
	for i := range cfg.text.NumHiddenLayers {
		if window > 0 && isSliding(i) {
			est.Layers.Sliding++
			est.Sliding += symmetricKV(heads, windowSlots(numCtx, window), dim, elem)
			continue
		}
		est.Layers.Attention++
		est.Attention += symmetricKV(heads, slots(numCtx), dim, elem)
	}
	return est
}
