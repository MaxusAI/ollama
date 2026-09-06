// Package kvsize estimates how many bytes a model's per-layer caches occupy at
// a given context length, reading nothing but the Hugging Face config.json (and
// the optional draft/config.json) that ship in the manifest.
//
// It exists so the parent process can price a context rung *before* it starts a
// runner: admission previously compared weights alone, so every num_ctx rung
// admitted identically and a rung that did not fit died mid-prefill instead of
// being refused (docs/maxusai/mlx-admission-prices-weights-only.md).
//
// The package is deliberately pure Go with no cgo: it must not import
// x/mlxrunner/mlx or x/models/..., because that would drag MLX into the server
// binary. The cost of that separation is that the rules here are a *copy* of the
// decisions the model packages make in their NewCaches methods, so every rule
// below names the code it mirrors. When a model package changes its cache
// layout, this package has to follow.
//
// Three cache kinds exist (x/mlxrunner/cache):
//
//   - cache.NewKVCache            grows with num_ctx in Step-sized blocks
//   - cache.NewRotatingKVCache(w) bounded by the sliding window w
//   - cache.NewRecurrentCache(..) constant, independent of num_ctx
//
// An architecture with no rule here reports Known == false. The caller must then
// fall back to its previous behaviour and admit; refusing on an unpriced model
// would be a regression. Under-pricing is the deliberate bias throughout:
// over-refusal on a serving host is worse than the over-admission we have today.
package kvsize

import (
	"encoding/json"
	"strings"
)

// Step mirrors the allocation step every attention cache uses
// (x/mlxrunner/cache/kvcache.go NewKVCache, rotating.go NewRotatingKVCache).
// A full cache's key/value buffers are grown in whole multiples of it.
const Step = 256

// LayerCounts breaks a model's cache-owning layers down by cache kind. Layers
// that own no cache at all (gemma4's KV-shared tail, nemotron's 'E'/'-' layers)
// appear in none of the three.
type LayerCounts struct {
	Attention int // cache.NewKVCache: grows with num_ctx
	Sliding   int // cache.NewRotatingKVCache: bounded by the window
	Recurrent int // cache.NewRecurrentCache and friends: constant
}

// Total is the number of layers that own a cache of any kind.
func (l LayerCounts) Total() int { return l.Attention + l.Sliding + l.Recurrent }

func (l *LayerCounts) add(o LayerCounts) {
	l.Attention += o.Attention
	l.Sliding += o.Sliding
	l.Recurrent += o.Recurrent
}

// Estimate is the cache footprint of one model (target plus its draft, if any)
// at NumCtx tokens. Byte fields are the sum over target and draft.
type Estimate struct {
	// Known is false when no rule matched the target architecture. Every byte
	// field is then zero and the caller must not refuse a load on it.
	Known bool
	// Arch is the target architecture the estimate dispatched on.
	Arch string
	// DraftArch is the draft architecture, empty when the model ships no
	// draft/config.json. DraftKnown is false when a draft exists but no rule
	// matched it, in which case its caches are missing from the totals.
	DraftArch  string
	DraftKnown bool

	NumCtx int
	// ElemBytes is the activation element width the caches are stored in: 2
	// for bf16/fp16 checkpoints, 4 for fp32. Weight quantization does not
	// change it -- caches hold activations, not weights.
	ElemBytes int

	Attention uint64 // full KV caches, grows with NumCtx
	Sliding   uint64 // rotating KV caches, bounded by the sliding window
	Recurrent uint64 // recurrent/conv state, constant in NumCtx

	Layers LayerCounts
}

// Total is the estimated cache footprint in bytes.
func (e Estimate) Total() uint64 { return e.Attention + e.Sliding + e.Recurrent }

func (e *Estimate) add(o Estimate) {
	e.Attention += o.Attention
	e.Sliding += o.Sliding
	e.Recurrent += o.Recurrent
	e.Layers.add(o.Layers)
}

// Model estimates the cache footprint of a model at numCtx tokens. config is
// the target's config.json; draft is draft/config.json, or nil when the model
// ships no draft. A numCtx of zero or less yields a zero estimate.
func Model(config, draft []byte, numCtx int) Estimate {
	if numCtx < 0 {
		numCtx = 0
	}

	cfg, err := parse(config)
	if err != nil {
		return Estimate{NumCtx: numCtx}
	}

	rule, arch := lookup(targetRules, cfg)
	est := Estimate{
		Arch:      arch,
		NumCtx:    numCtx,
		ElemBytes: cfg.elemBytes(),
	}
	if rule == nil {
		return est
	}
	priced, ok := rule(cfg, numCtx)
	if !ok {
		// The architecture is known but this config is not shaped the way the
		// rule needs (a missing hybrid pattern, no layers). Report it unknown
		// so the caller falls back rather than pricing a guess.
		return est
	}
	est.Known = true
	est.add(priced)

	if len(draft) == 0 {
		return est
	}
	dcfg, err := parse(draft)
	if err != nil {
		return est
	}
	drule, darch := lookup(draftRules, dcfg)
	est.DraftArch = darch
	if drule == nil {
		// A draft we cannot price is left out of the totals rather than
		// refusing the whole estimate: under-pricing keeps today's
		// over-admission, over-refusal is the failure mode to avoid.
		return est
	}
	draftPriced, ok := drule(dcfg, numCtx)
	if !ok {
		return est
	}
	est.DraftKnown = true
	est.add(draftPriced)
	return est
}

// rule prices one config's caches. numCtx is the context rung being admitted.
// It reports false when the config does not carry what the rule needs, which
// makes the whole estimate unknown rather than wrong.
type rule func(cfg *config, numCtx int) (Estimate, bool)

func lookup(table map[string]rule, cfg *config) (rule, string) {
	for _, key := range []string{cfg.Arch, cfg.ModelType} {
		if key == "" {
			continue
		}
		if r, ok := table[key]; ok {
			return r, key
		}
	}
	if cfg.Arch != "" {
		return nil, cfg.Arch
	}
	return nil, cfg.ModelType
}

// slots is how many token slots a full attention cache holds after numCtx
// tokens. cache.KVCache.appendKV grows the buffer in whole Step blocks
// (`steps := (c.step + L - 1) / c.step; Zeros(..., steps*c.step, ...)`), so the
// buffer a full prompt leaves behind is num_ctx rounded up to a Step multiple.
func slots(numCtx int) int {
	if numCtx <= 0 {
		return 0
	}
	return ((numCtx + Step - 1) / Step) * Step
}

// windowSlots is how many token slots a rotating cache holds. RotatingKVCache's
// decode path grows by min(step, maxSize-prev) and never past maxSize
// (x/mlxrunner/cache/rotating.go update), so the buffer is the window or the
// rounded context, whichever is smaller.
//
// The batched prefill path (rotating.go concat) transiently exceeds this: it
// trims to maxSize-1 and concatenates the whole chunk, so a chunk of
// prefillChunkSize tokens holds up to maxSize-1+chunk slots for the duration of
// that forward. That excess is deliberately NOT priced here -- it is a transient
// the admission headroom covers, and pricing it would over-refuse every sliding
// model. See docs/maxusai/adr/0034 and the task note.
func windowSlots(numCtx, window int) int {
	if window <= 0 {
		return slots(numCtx)
	}
	return min(slots(numCtx), window)
}

// attnBytes is the byte size of one attention cache's key and value buffers.
// KVCache allocates keys [B=1, H, T, Dk] and values [B=1, H, T, Dv], taking H
// from the keys, so a value width that differs from the key width (MLA's zero,
// QSA's 3-wide positions) is expressed as its own dim and element size.
func attnBytes(heads, tokens, keyDim, keyElem, valueDim, valueElem int) uint64 {
	if heads <= 0 || tokens <= 0 {
		return 0
	}
	perToken := uint64(max(keyDim, 0))*uint64(keyElem) + uint64(max(valueDim, 0))*uint64(valueElem)
	return uint64(heads) * uint64(tokens) * perToken
}

// symmetricKV is the common case: keys and values share the head count, width
// and element size.
func symmetricKV(heads, tokens, dim, elem int) uint64 {
	return attnBytes(heads, tokens, dim, elem, dim, elem)
}

// recurrentBytes mirrors cache.RecurrentCache.Get
// (x/mlxrunner/cache/recurrent.go:117-118): a conv state [1, convTail, convDim]
// in the activation dtype plus a delta state
// [1, numVHeads, headVDim, headKDim] that is always float32. Neither depends on
// num_ctx.
func recurrentBytes(convTail, convDim, numVHeads, headVDim, headKDim, elem int) uint64 {
	conv := uint64(max(convTail, 0)) * uint64(max(convDim, 0)) * uint64(elem)
	delta := uint64(max(numVHeads, 0)) * uint64(max(headVDim, 0)) * uint64(max(headKDim, 0)) * 4
	return conv + delta
}

// config is the subset of a Hugging Face config.json this package reads. The
// model packages parse the same fields; the JSON tags here must match theirs.
type config struct {
	Arch      string
	ModelType string
	// text is the section the model packages treat as the text config:
	// text_config when nested (gemma4, qwen3.5 VL), llm_config for
	// nemotron_h, the document root otherwise.
	text textConfig
}

type textConfig struct {
	Dtype      string `json:"dtype"`
	TorchDtype string `json:"torch_dtype"`

	NumHiddenLayers   int      `json:"num_hidden_layers"`
	NumAttentionHeads int      `json:"num_attention_heads"`
	NumKeyValueHeads  int      `json:"num_key_value_heads"`
	HiddenSize        int      `json:"hidden_size"`
	HeadDim           int      `json:"head_dim"`
	LayerTypes        []string `json:"layer_types"`
	SlidingWindow     int      `json:"sliding_window"`

	// gemma4
	GlobalHeadDim          int  `json:"global_head_dim"`
	NumGlobalKeyValueHeads int  `json:"num_global_key_value_heads"`
	AttentionKEqV          bool `json:"attention_k_eq_v"`
	NumKVSharedLayers      int  `json:"num_kv_shared_layers"`
	SlidingWindowPattern   int  `json:"sliding_window_pattern"`

	// cohere2_moe
	PrefixDenseSlidingWindowPattern int `json:"prefix_dense_sliding_window_pattern"`
	FirstKDenseReplace              int `json:"first_k_dense_replace"`

	// qwen3_5 / qwen3_5_moe / qwen4_exp linear (recurrent) layers
	FullAttentionInterval int `json:"full_attention_interval"`
	LinearConvKernelDim   int `json:"linear_conv_kernel_dim"`
	LinearNumKeyHeads     int `json:"linear_num_key_heads"`
	LinearNumValueHeads   int `json:"linear_num_value_heads"`
	LinearKeyHeadDim      int `json:"linear_key_head_dim"`
	LinearValueHeadDim    int `json:"linear_value_head_dim"`

	// qwen4_exp QSA side caches and engram cache
	IndexerHeadDim    int `json:"indexer_head_dim"`
	IndexerKVHeads    int `json:"indexer_kv_heads"`
	NGramSize         int `json:"ngram_size"`
	PLEConvKernelSize int `json:"ple_conv_kernel_size"`
	HCCount           int `json:"hc_count"`

	// nemotron_h
	HybridOverridePattern string `json:"hybrid_override_pattern"`
	ConvKernel            int    `json:"conv_kernel"`
	SSMStateSize          int    `json:"ssm_state_size"`
	MambaNumHeads         int    `json:"mamba_num_heads"`
	MambaHeadDim          int    `json:"mamba_head_dim"`
	NGroups               int    `json:"n_groups"`

	// glm4_moe_lite (MLA)
	KVLoraRank    int `json:"kv_lora_rank"`
	QKRopeHeadDim int `json:"qk_rope_head_dim"`
}

func parse(data []byte) (*config, error) {
	var envelope struct {
		Architectures []string        `json:"architectures"`
		ModelType     string          `json:"model_type"`
		TextConfig    json.RawMessage `json:"text_config"`
		LLMConfig     json.RawMessage `json:"llm_config"`
		Dtype         string          `json:"dtype"`
		TorchDtype    string          `json:"torch_dtype"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return nil, err
	}

	cfg := &config{ModelType: envelope.ModelType}
	if len(envelope.Architectures) > 0 {
		cfg.Arch = envelope.Architectures[0]
	}

	// text_config for gemma4 and the qwen3.5 family, llm_config for
	// nemotron_h (x/models/nemotron_h/nemotron_h.go configEnvelope), the
	// root for everything else.
	section := data
	for _, nested := range []json.RawMessage{envelope.TextConfig, envelope.LLMConfig} {
		if len(nested) > 0 && string(nested) != "null" {
			section = nested
			break
		}
	}
	if err := json.Unmarshal(section, &cfg.text); err != nil {
		return nil, err
	}
	// A nested section usually carries its own dtype; fall back to the root's
	// when it does not.
	if cfg.text.Dtype == "" {
		cfg.text.Dtype = envelope.Dtype
	}
	if cfg.text.TorchDtype == "" {
		cfg.text.TorchDtype = envelope.TorchDtype
	}
	return cfg, nil
}

// elemBytes is the width of one cached activation element. Caches store
// activations, so a quantized checkpoint still caches in its compute dtype;
// only an fp32 checkpoint doubles the width.
func (c *config) elemBytes() int {
	for _, d := range []string{c.text.TorchDtype, c.text.Dtype} {
		switch strings.ToLower(strings.TrimSpace(d)) {
		case "float32", "fp32", "float":
			return 4
		}
	}
	return 2
}

// layerType reports the entry of layer_types for layer i, lowercased, or "" if
// the config carries no usable layer_types.
func (c *config) layerType(i int) string {
	if len(c.text.LayerTypes) != c.text.NumHiddenLayers || i >= len(c.text.LayerTypes) {
		return ""
	}
	return strings.ToLower(c.text.LayerTypes[i])
}
