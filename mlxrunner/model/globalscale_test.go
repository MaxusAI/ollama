package model

import (
	"math"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

// LoadGlobalScale normalises shape and dtype and keeps the checkpoint's value;
// PrepareGatherQMMGlobalScale only broadcasts what it is given. Storing m rather
// than MLX's m*Nvfp4MaxProduct is ADR 0039, and the round-trip test below is why.
func TestGlobalScaleConversion(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if got := LoadGlobalScale(nil); got != nil {
			t.Fatal("nil scale did not convert to nil")
		}
		if got := PrepareGatherQMMGlobalScale(nil, 4); got != nil {
			t.Fatal("nil scale did not broadcast to nil")
		}

		// Checkpoints ship a scalar as either [] or [1]; both normalize to [1]
		// so a mix of the two can still be stacked.
		for _, scalar := range []*mlx.Array{
			mlx.NewScalarArray(2),
			mlx.FromValues([]float32{2}, 1),
		} {
			got := LoadGlobalScale(scalar)
			mlx.Eval(got)
			if dims := got.Dims(); len(dims) != 1 || dims[0] != 1 {
				t.Fatalf("converted scalar dims = %v, want [1]", dims)
			}
			if want := float32(2); got.Floats()[0] != want {
				t.Fatalf("stored scalar = %v, want the checkpoint's own %v", got.Floats()[0], want)
			}
		}

		checkpointScales := []float32{0.5, 1, 2, 4}
		perExpert := LoadGlobalScale(mlx.FromValues(checkpointScales, 4))
		mlx.Eval(perExpert)
		for i, got := range perExpert.Floats() {
			if want := checkpointScales[i]; got != want {
				t.Fatalf("stored scale[%d] = %v, want the checkpoint's own %v", i, got, want)
			}
		}

		// Broadcasting is value-preserving, and does not convert a second time.
		bank := PrepareGatherQMMGlobalScale(LoadGlobalScale(mlx.FromValues([]float32{2}, 1)), 4)
		mlx.Eval(bank)
		if dims := bank.Dims(); len(dims) != 1 || dims[0] != 4 {
			t.Fatalf("bank dims = %v, want [4]", dims)
		}
		for i, got := range bank.Floats() {
			if want := float32(2); got != want {
				t.Fatalf("bank[%d] = %v, want %v", i, got, want)
			}
		}

		// A scale count that is neither a checkpoint-wide scalar nor one per
		// expert is malformed and fails the load instead of being ignored.
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("mismatched scale count did not fail")
				}
			}()
			PrepareGatherQMMGlobalScale(LoadGlobalScale(mlx.FromValues([]float32{1, 2}, 2)), 4)
		}()
	})
}

func TestSameGlobalScales(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		perExpert := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{0.5, 1, 2, 4}, 4), 4)
		same := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{0.5, 1, 2, 4}, 4), 4)
		reordered := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{4, 2, 1, 0.5}, 4), 4)
		broadcast := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{2}, 1), 4)
		shorter := PrepareGatherQMMGlobalScale(mlx.FromValues([]float32{2}, 1), 2)

		for _, tt := range []struct {
			name string
			a, b *mlx.Array
			want bool
		}{
			{"both nil", nil, nil, true},
			{"only one nil", perExpert, nil, false},
			{"identical array", perExpert, perExpert, true},
			{"equal values", perExpert, same, true},
			{"same values reordered", perExpert, reordered, false},
			{"broadcast scalar matches itself", broadcast, broadcast, true},
			{"different expert counts", broadcast, shorter, false},
		} {
			if got := SameGlobalScales(tt.a, tt.b); got != tt.want {
				t.Fatalf("%s: SameGlobalScales() = %v, want %v", tt.name, got, tt.want)
			}
		}
	})
}

// TestGatherQMMGlobalScaleMatchesDequantized checks the whole scale chain —
// Prepare's conversion to amax units plus the GatherQMM wrapper — against the
// dequantized weights gathered densely. Metal runs the native kernel path;
// CUDA runs the wrapper's output scaling. ModelOpt exports a checkpoint-wide
// scalar, so both it and the per-expert form are covered.
func TestGatherQMMGlobalScaleMatchesDequantized(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() && !mlx.CUDAIsAvailable() {
			t.Skip("gather_qmm requires a GPU backend")
		}

		const experts, rows, cols, groupSize = 4, 32, 64, 16
		packed := make([]uint32, experts*rows*cols/8)
		for i := range packed {
			for j := range 8 {
				packed[i] |= uint32((i*8+j)%16) << (4 * j)
			}
		}
		scales := make([]uint8, experts*rows*cols/groupSize)
		for i := range scales {
			scales[i] = uint8(((i % 4) + 6) << 3)
		}
		weights := mlx.FromValues(packed, experts, rows, cols/8)
		blockScales := mlx.FromValues(scales, experts, rows, cols/groupSize)

		xValues := make([]float32, cols)
		for i := range xValues {
			xValues[i] = float32(i%7-3) / 8
		}
		x := mlx.FromValues(xValues, 1, cols).AsType(mlx.DTypeBFloat16)
		indices := mlx.FromValues([]int32{0, 1, 2, 3}, 1, experts)

		for _, checkpointScale := range [][]float32{
			{0.5, 1, 2, 4}, // one scale per expert
			{2},            // checkpoint-wide scalar
		} {
			checkpoint := mlx.FromValues(checkpointScale, len(checkpointScale))
			kernelScale := PrepareGatherQMMGlobalScale(checkpoint, experts)

			got := mlx.GatherQMM(
				x, weights, blockScales, nil, nil, indices,
				true, groupSize, 4, "nvfp4", kernelScale, false,
			).AsType(mlx.DTypeFloat32)
			dense := mlx.Dequantize(
				weights, blockScales, nil, groupSize, 4, "nvfp4", checkpoint,
			).AsType(mlx.DTypeFloat32)
			want := mlx.GatherMM(
				x.AsType(mlx.DTypeFloat32), mlx.Transpose(dense, 0, 2, 1), nil, indices, false,
			)
			mlx.Eval(got, want)

			gotValues, wantValues := got.Floats(), want.Floats()
			if len(gotValues) != len(wantValues) {
				t.Fatalf("result length = %d, want %d", len(gotValues), len(wantValues))
			}
			for i := range gotValues {
				if math.IsNaN(float64(gotValues[i])) || math.IsInf(float64(gotValues[i]), 0) {
					t.Fatalf("result[%d] = %v, want finite", i, gotValues[i])
				}
				delta := math.Abs(float64(gotValues[i] - wantValues[i]))
				tolerance := 0.01 * math.Max(math.Abs(float64(wantValues[i])), 1)
				if delta > tolerance {
					t.Fatalf("result[%d] = %v, want %v (delta %v > %v)", i, gotValues[i], wantValues[i], delta, tolerance)
				}
			}
		}
	})
}

// One key policy for every model. The canonical name import writes wins;
// ModelOpt's own name is the fallback for un-imported checkpoints; and an
// activation scale is consumed but never mistaken for a weight scale.
func TestReadGlobalScale(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		scale := func(v float32) *mlx.Array { return mlx.FromValues([]float32{v}, 1) }

		for _, tt := range []struct {
			name     string
			tensors  map[string]*mlx.Array
			want     float32 // 0 means "expect nil"
			consumed []string
		}{
			{
				name:     "canonical name",
				tensors:  map[string]*mlx.Array{"w.weight.global_scale": scale(0.5)},
				want:     0.5,
				consumed: []string{"w.weight.global_scale"},
			},
			{
				name:     "modelopt fallback",
				tensors:  map[string]*mlx.Array{"w.weight_scale_2": scale(0.25)},
				want:     0.25,
				consumed: []string{"w.weight_scale_2"},
			},
			{
				name: "canonical wins over fallback",
				tensors: map[string]*mlx.Array{
					"w.weight.global_scale": scale(0.5),
					"w.weight_scale_2":      scale(0.25),
				},
				want:     0.5,
				consumed: []string{"w.weight.global_scale", "w.weight_scale_2"},
			},
			{
				// An activation scale is not a weight scale. Picking it up
				// would silently rescale every output.
				name:     "activation scale is never the weight scale",
				tensors:  map[string]*mlx.Array{"w.weight.input_global_scale": scale(8)},
				want:     0,
				consumed: []string{"w.weight.input_global_scale"},
			},
			{
				name: "activation scale alongside a weight scale",
				tensors: map[string]*mlx.Array{
					"w.weight.global_scale":       scale(0.5),
					"w.weight.input_global_scale": scale(8),
				},
				want:     0.5,
				consumed: []string{"w.weight.global_scale", "w.weight.input_global_scale"},
			},
			{name: "absent", tensors: map[string]*mlx.Array{}, want: 0},
		} {
			got, consumed := ReadGlobalScale(tt.tensors, "w.weight")
			if tt.want == 0 {
				if got != nil {
					mlx.Eval(got)
					t.Fatalf("%s: got %v, want nil", tt.name, got.Floats())
				}
			} else {
				if got == nil {
					t.Fatalf("%s: got nil, want [%v]", tt.name, tt.want)
				}
				mlx.Eval(got)
				if values := got.Floats(); len(values) != 1 || values[0] != tt.want {
					t.Fatalf("%s: got %v, want [%v]", tt.name, values, tt.want)
				}
			}
			if len(consumed) != len(tt.consumed) {
				t.Fatalf("%s: consumed %v, want %v", tt.name, consumed, tt.consumed)
			}
			for _, key := range tt.consumed {
				if !slices.Contains(consumed, key) {
					t.Fatalf("%s: consumed %v, missing %q", tt.name, consumed, key)
				}
			}
		}
	})
}

// The defect ADR 0039 removes: storing MLX's m*Nvfp4MaxProduct meant every wrapper
// that applies the scale itself divided it back out, and that round trip is not the
// identity in float32. These are real global scales from the gemma4 checkpoints --
// 17 of 31b's 191 vision scales lose a ulp, which 27 encoder layers grow into the
// golden delta issue #312 chased. Stored as the checkpoint's own value, every one
// survives bit-exactly, and the boundary conversion is applied once where MLX reads it.
func TestGlobalScaleSurvivesStorageBitExactly(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		// Values chosen so the old round trip demonstrably failed on some of them.
		checkpoint := []float32{
			0.0013580322265625, 0.00136566162109375, 0.001373291015625,
			0.0069580078125, 0.007080078125, 0.00732421875, 1, 0.5, 2, 3,
		}
		stored := LoadGlobalScale(mlx.FromValues(checkpoint, len(checkpoint)))
		mlx.Eval(stored)
		for i, got := range stored.Floats() {
			if got != checkpoint[i] {
				t.Fatalf("stored scale[%d] = %v, want %v bit-exactly", i, got, checkpoint[i])
			}
		}

		// The old convention, for the record: convert to MLX's form and back.
		roundTripped := mlx.DivScalar(mlx.MulScalar(stored, mlx.Nvfp4MaxProduct), mlx.Nvfp4MaxProduct)
		mlx.Eval(roundTripped)
		lost := 0
		for i, got := range roundTripped.Floats() {
			if got != checkpoint[i] {
				lost++
			}
			_ = i
		}
		if lost == 0 {
			t.Skip("this float32 round trip happens to be exact for every value here; " +
				"the rule still holds, the fixture just does not witness it")
		}
		t.Logf("the old round trip lost %d of %d scales; the stored convention loses none", lost, len(checkpoint))
	})
}

// The boundary conversion is the only place the MLX form appears, and it is exact.
func TestToMLXRepresentationIsTheBoundary(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if got := mlx.ToMLXRepresentation(nil); got != nil {
			t.Fatal("nil scale did not convert to nil")
		}
		stored := LoadGlobalScale(mlx.FromValues([]float32{1, 2, 0.5}, 3))
		converted := mlx.ToMLXRepresentation(stored)
		mlx.Eval(converted)
		for i, want := range []float32{1, 2, 0.5} {
			if got := converted.Floats()[i]; got != want*float32(mlx.Nvfp4MaxProduct) {
				t.Fatalf("converted[%d] = %v, want %v", i, got, want*float32(mlx.Nvfp4MaxProduct))
			}
		}
		// GatherQMM's identity leaves a bank unscaled: 1 stored, Nvfp4MaxProduct at MLX.
		id := GatherQMMIdentityScale()
		mlx.Eval(id)
		if got := id.Floats()[0]; got != 1 {
			t.Fatalf("stored identity = %v, want 1", got)
		}
		atMLX := mlx.ToMLXRepresentation(id)
		mlx.Eval(atMLX)
		if got := atMLX.Floats()[0]; got != float32(mlx.Nvfp4MaxProduct) {
			t.Fatalf("identity at the MLX boundary = %v, want %v", got, float32(mlx.Nvfp4MaxProduct))
		}
	})
}
