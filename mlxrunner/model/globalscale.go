package model

import "github.com/ollama/ollama/mlx"

// Import rewrites every vendor spelling to ".global_scale"; "_scale_2" is
// ModelOpt's own name, reached when a checkpoint skips import.
var globalScaleSuffixes = []string{".global_scale", "_scale_2"}

// These scale the activations, never the weight, but are freed alongside it.
var activationScaleSuffixes = []string{".input_global_scale", ".input_scale"}

// ReadGlobalScale returns a weight's NVFP4 global scale in MLX's
// representation, and the companion keys the caller should release. Candidate
// keys are tried in order, so pass the resolved tensor key before any model.
func ReadGlobalScale(tensors map[string]*mlx.Array, weightKeys ...string) (*mlx.Array, []string) {
	var found *mlx.Array
	var consumed []string
	for _, key := range weightKeys {
		if key == "" {
			continue
		}
		for _, suffix := range globalScaleSuffixes {
			scale, ok := tensors[key+suffix]
			if !ok || scale == nil {
				continue
			}
			if found == nil {
				found = scale
			}
			consumed = append(consumed, key+suffix)
		}
		for _, suffix := range activationScaleSuffixes {
			if _, ok := tensors[key+suffix]; ok {
				consumed = append(consumed, key+suffix)
			}
		}
	}
	return LoadGlobalScale(found), consumed
}

// LoadGlobalScale normalises a checkpoint multiplier for storage: float32, and
// flattened, because a scalar ships as either [] or [1] and stacking a mix of the
// two fails. The VALUE is the checkpoint's own m, unchanged.
//
// It is deliberately not MLX's m*Nvfp4MaxProduct form. Storing that meant every
// wrapper applying the scale itself had to divide it back out, and that round trip
// is not the identity in float32 -- one ulp on 17 of 31b's 191 vision scales, which
// 27 encoder layers grow into a visible golden delta (issue #312). Conversion now
// happens at the two places MLX consumes a scale, via mlx.ToMLXRepresentation
// (ADR 0039).
func LoadGlobalScale(globalScale *mlx.Array) *mlx.Array {
	if globalScale == nil {
		return nil
	}
	return mlx.Reshape(globalScale.AsType(mlx.DTypeFloat32), int32(globalScale.Size()))
}

// PrepareGatherQMMGlobalScale broadcasts an already-converted global scale
// into the one-entry-per-expert bank gather_qmm wants. Materialized dense: the
// kernel indexes it by raw offset, and a broadcast view is one element of
// storage.
func PrepareGatherQMMGlobalScale(globalScale *mlx.Array, numExperts int) *mlx.Array {
	if globalScale == nil {
		return nil
	}
	return mlx.Contiguous(mlx.BroadcastTo(globalScale, int32(numExperts)), false)
}

// GatherQMMIdentityScale is the scale that leaves an expert bank unscaled, for
// rows folded into a scaled bank without a scale of their own. In the stored
// convention that is 1, not Nvfp4MaxProduct: the conversion happens where the bank
// reaches MLX (ADR 0039).
func GatherQMMIdentityScale() *mlx.Array {
	return mlx.FromValues([]float32{1}, 1)
}

// SameGlobalScales reports whether two prepared banks hold the same scale for
// every expert, which is what lets two projections share one fused bank.
func SameGlobalScales(a, b *mlx.Array) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	if a == b {
		return true
	}
	if a.Size() != b.Size() {
		return false
	}
	mlx.Eval(a, b)
	aValues, bValues := a.Floats(), b.Floats()
	for i := range aValues {
		if aValues[i] != bValues[i] {
			return false
		}
	}
	return true
}
