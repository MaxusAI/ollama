package nn

import (
	"os"
	"strconv"
	"strings"
	"sync"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// Layer is the interface for neural network layers with a Forward method.
type Layer interface {
	Forward(x *mlx.Array) *mlx.Array
}

// LinearLayer is an interface for linear layers (both regular and quantized).
type LinearLayer interface {
	Forward(x *mlx.Array) *mlx.Array
	OutputDim() int32
}

// EmbeddingLayer is an interface for embedding layers that can also expose a
// tied-output projection when the model reuses embedding weights as the LM head.
type EmbeddingLayer interface {
	Forward(indices *mlx.Array) *mlx.Array
	AsLinear() LinearLayer
}

// Conv1d applies 1D convolution over NLC input.
type Conv1d struct {
	Weight   *mlx.Array
	Bias     *mlx.Array
	Stride   int32
	Padding  int32
	Dilation int32
	Groups   int32
}

func NewConv1d(weight, bias *mlx.Array, stride, padding, dilation, groups int32) *Conv1d {
	if stride <= 0 {
		stride = 1
	}
	if dilation <= 0 {
		dilation = 1
	}
	if groups <= 0 {
		groups = 1
	}
	return &Conv1d{
		Weight:   weight,
		Bias:     bias,
		Stride:   stride,
		Padding:  padding,
		Dilation: dilation,
		Groups:   groups,
	}
}

func (c *Conv1d) Forward(x *mlx.Array) *mlx.Array {
	return mlx.Conv1d(x, c.Weight, c.Bias, c.Stride, c.Padding, c.Dilation, c.Groups)
}

// Linear applies an affine transformation: y = x @ W.T + b
type Linear struct {
	Weight *mlx.Array
	Bias   *mlx.Array
}

func NewLinear(weight *mlx.Array, bias *mlx.Array) *Linear {
	if bias != nil && bias.Valid() && bias.DType() != weight.DType() {
		bias = bias.AsType(weight.DType())
	}
	return &Linear{Weight: weight, Bias: bias}
}

func (l *Linear) Forward(x *mlx.Array) *mlx.Array {
	w := l.Weight.Transpose(1, 0)
	if l.Bias != nil && l.Bias.Valid() {
		return l.Bias.Addmm(x, w, 1.0, 1.0)
	}
	return x.Matmul(w)
}

func (l *Linear) OutputDim() int32 {
	return int32(l.Weight.Dim(0))
}

// QuantizedLinear applies an affine transformation using quantized weights.
type QuantizedLinear struct {
	Weight      *mlx.Array // Quantized weight data
	Scales      *mlx.Array // Scale factors for dequantization
	QBiases     *mlx.Array // Quantization biases (nil for nvfp4)
	Bias        *mlx.Array // Layer bias [output_dims] or nil
	GlobalScale *mlx.Array // Per-tensor or per-row global scale for double-scale nvfp4 (nil for standard)
	GroupSize   int
	Bits        int
	Mode        string
}

func NewQuantizedLinear(weight *mlx.Array, bias *mlx.Array, groupSize, bits int, mode string) *QuantizedLinear {
	qw, scales, qbiases := mlx.Quantize(weight, groupSize, bits, mode)
	if qbiases != nil {
		mlx.Eval(qw, scales, qbiases)
	} else {
		mlx.Eval(qw, scales)
	}
	if bias != nil && bias.Valid() && bias.DType() != weight.DType() {
		bias = bias.AsType(weight.DType())
	}
	return &QuantizedLinear{
		Weight:    qw,
		Scales:    scales,
		QBiases:   qbiases,
		Bias:      bias,
		GroupSize: groupSize,
		Bits:      bits,
		Mode:      mode,
	}
}

var quantizedLinearOutputScale = mlx.Compile2(
	"QuantizedLinearOutputScale",
	func(out, scale *mlx.Array) *mlx.Array {
		return mlx.Mul(out, scale).AsType(out.DType())
	},
	mlx.Shapeless(),
)

// PrefillDequantRows is the row count at or above which a nvfp4 QuantizedLinear
// on a CUDA device dequantises its weights to the activation dtype and runs the
// dense GEMM (cuBLASLt) instead of MLX's mixed-input kernel. On the RTX PRO 6000
// the mixed-input kernel wins below ~512 rows (decode, small chunks) and loses
// 1.4-2x to the dense path from ~2048 rows, the prefill chunk; a real image
// prefill gains 1.2-2.1x (docs/maxusai/tasks/mlx-prefill-dequant-gemm.md).
//
// It costs memory: MLX evaluates a chunk's forward pass as one graph, so the
// dequantised copies of many layers are live at once -- measured at up to
// +6.8 GiB on gemma4:31b. Admission does not price that, so this stays off by
// default and must not be enabled on a shared card until the copies are
// bounded. 0, the default, keeps the mixed-input kernel everywhere. Opt in with
// OLLAMA_MLX_PREFILL_DEQUANT_ROWS.
var PrefillDequantRows = prefillDequantRowsFromEnv(os.Getenv("OLLAMA_MLX_PREFILL_DEQUANT_ROWS"))

// prefillDequantRowsFromEnv parses the opt-in; anything that is not a
// non-negative integer means off.
func prefillDequantRowsFromEnv(v string) int {
	n, err := strconv.Atoi(strings.TrimSpace(v))
	if err != nil || n < 0 {
		return 0
	}
	return n
}

var cudaAvailable = sync.OnceValue(func() bool { return mlx.CUDAIsAvailable() })

// useDequantGEMM is the policy behind PrefillDequantRows: only when opted in,
// only nvfp4 (the measured mode), only on CUDA (Metal's kernels were not
// measured), only at or above the row threshold.
func useDequantGEMM(rows, threshold int, mode string, cuda bool) bool {
	return threshold > 0 && cuda && mode == "nvfp4" && rows >= threshold
}

// rows is the number of activation rows x carries: everything but the last dim.
func rows(x *mlx.Array) int {
	k := x.Dim(x.NumDims() - 1)
	if k <= 0 {
		return 0
	}
	return x.Size() / k
}

// denseGEMM dequantises the layer to x's dtype and multiplies on the dense
// path: the same weights and bf16 math as the mixed-input kernel with a
// different accumulation order, and a transient [N, K] copy of the layer that
// lives only inside this call's graph.
func (ql *QuantizedLinear) denseGEMM(x *mlx.Array) *mlx.Array {
	w := mlx.Dequantize(ql.Weight, ql.Scales, ql.QBiases, ql.GroupSize, ql.Bits, ql.Mode, nil)
	if w.DType() != x.DType() {
		w = w.AsType(x.DType())
	}
	return x.Matmul(w.Transpose(1, 0))
}

func (ql *QuantizedLinear) Forward(x *mlx.Array) *mlx.Array {
	var out *mlx.Array
	if useDequantGEMM(rows(x), PrefillDequantRows, ql.Mode, cudaAvailable()) {
		out = ql.denseGEMM(x)
	} else {
		out = mlx.QuantizedMatmul(x, ql.Weight, ql.Scales, ql.QBiases, true, ql.GroupSize, ql.Bits, ql.Mode)
	}
	if ql.GlobalScale != nil {
		// Double-scale nvfp4 (e.g., NVIDIA ModelOpt): standard quantized_matmul
		// followed by global_scale multiply. The global_scale is F32, per-tensor
		// (weight_scale_2 in NVIDIA's format) or per-row.
		// TODO: switch to a fused double-scale matmul once MLX has kernel
		// coverage for this path.
		out = quantizedLinearOutputScale(out, ql.GlobalScale)
	}
	if ql.Bias != nil && ql.Bias.Valid() {
		bias := ql.Bias
		if bias.DType() != out.DType() {
			bias = bias.AsType(out.DType())
		}
		out = out.Add(bias)
	}
	return out
}

func (ql *QuantizedLinear) OutputDim() int32 {
	return int32(ql.Weight.Dim(0))
}

// RMSNorm represents an RMS normalization layer.
type RMSNorm struct {
	Weight *mlx.Array
	Eps    float32
}

func NewRMSNorm(weight *mlx.Array, eps float32) *RMSNorm {
	return &RMSNorm{Weight: weight, Eps: eps}
}

func (rn *RMSNorm) Forward(x *mlx.Array, eps float32) *mlx.Array {
	if eps == 0 {
		eps = rn.Eps
	}
	return mlx.RMSNormFn(x, rn.Weight, eps)
}

// Embedding represents an embedding layer.
type Embedding struct {
	Weight *mlx.Array
}

func NewEmbedding(weight *mlx.Array) *Embedding {
	return &Embedding{Weight: weight}
}

func (e *Embedding) Forward(indices *mlx.Array) *mlx.Array {
	return e.Weight.TakeAxis(indices, 0)
}

func (e *Embedding) AsLinear() LinearLayer {
	return NewLinear(e.Weight, nil)
}

// QuantizedEmbedding performs row-wise embedding lookup from affine/nvfp4/etc.
// packed weights and dequantizes only the selected rows.
type QuantizedEmbedding struct {
	Weight      *mlx.Array
	Scales      *mlx.Array
	QBiases     *mlx.Array
	GlobalScale *mlx.Array // Per-tensor global scale for double-scale nvfp4 (nil for standard)
	GroupSize   int
	Bits        int
	Mode        string
}

func (qe *QuantizedEmbedding) Forward(indices *mlx.Array) *mlx.Array {
	weight := qe.Weight.TakeAxis(indices, 0)
	scales := qe.Scales.TakeAxis(indices, 0)
	var qbiases *mlx.Array
	if qe.QBiases != nil && qe.QBiases.Valid() {
		qbiases = qe.QBiases.TakeAxis(indices, 0)
	}
	return mlx.Dequantize(weight, scales, qbiases, qe.GroupSize, qe.Bits, qe.Mode, qe.GlobalScale)
}

func (qe *QuantizedEmbedding) AsLinear() LinearLayer {
	return &QuantizedLinear{
		Weight:      qe.Weight,
		Scales:      qe.Scales,
		QBiases:     qe.QBiases,
		GlobalScale: qe.GlobalScale,
		GroupSize:   qe.GroupSize,
		Bits:        qe.Bits,
		Mode:        qe.Mode,
	}
}

// LayerNorm represents a standard layer normalization layer (with bias).
type LayerNorm struct {
	Weight *mlx.Array
	Bias   *mlx.Array
	Eps    float32
}

func (ln *LayerNorm) Forward(x *mlx.Array) *mlx.Array {
	eps := ln.Eps
	if eps == 0 {
		eps = 1e-5
	}
	return mlx.LayerNormFn(x, ln.Weight, ln.Bias, eps)
}

// MultiLinearLayer is an interface for per-head linear layers.
type MultiLinearLayer interface {
	Forward(x *mlx.Array) *mlx.Array
}

// MultiLinear performs per-head linear projections.
// Weight shape: [num_heads, output_dims, input_dims]
type MultiLinear struct {
	Weight *mlx.Array
}

func NewMultiLinear(weight *mlx.Array) *MultiLinear {
	return &MultiLinear{Weight: weight}
}

func (ml *MultiLinear) Forward(x *mlx.Array) *mlx.Array {
	wT := ml.Weight.Transpose(0, 2, 1)
	return x.Matmul(wT)
}

// ApplyCausalMask applies causal (lower triangular) mask to attention scores.
func ApplyCausalMask(scores *mlx.Array) *mlx.Array {
	shape := scores.Dims()
	seqLen := int32(shape[2])
	mask := mlx.Tri(seqLen, seqLen, 0)
	negInf := mlx.NewScalarArray(float32(-1e9))
	mask = mask.ExpandDims(0).ExpandDims(0)
	return mlx.Where(mask, scores, negInf)
}

// ApplyCausalMaskWithOffset applies causal mask for cached attention.
func ApplyCausalMaskWithOffset(scores *mlx.Array, offset int32) *mlx.Array {
	if offset == 0 {
		return ApplyCausalMask(scores)
	}
	shape := scores.Dims()
	queryLen := int32(shape[2])
	keyLen := int32(shape[3])
	mask := mlx.Tri(queryLen, keyLen, int(offset))
	negInf := mlx.NewScalarArray(float32(-1e9))
	mask = mask.ExpandDims(0).ExpandDims(0)
	return mlx.Where(mask, scores, negInf)
}
