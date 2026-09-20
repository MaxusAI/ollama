package mlx

import (
	"math"
	"math/rand"
	"testing"

	"github.com/ollama/ollama/mlx/mlxthread/mlxthreadtest"
)

// TestFpQmmTKMod32 guards upstream MLX #3912: fp_qmm_t misread past the K
// dimension when K mod 32 == 16, silently corrupting most of the output.
//
// WHY THIS K. 4304 is the vision tower's mlp.down_proj contraction dimension in
// the gemma4 26b/31b nvfp4 checkpoints — stored packed as [1152, 538] at 8
// values per word, cross-checked by the group-16 scale tensor at [1152, 269].
// 12b carries no nvfp4 vision down_proj at all (its text ones are K = 15360 and
// 65536, both mod 32 == 0), which is why it was unaffected. 4288 is the nearest
// mod-32-aligned K and is the control: it must stay clean on every build, so a
// failure at 4288 means something other than this bug.
//
// WHAT IT LOOKED LIKE IN PRODUCTION. Not a crash and not a wrong answer that
// anyone would notice — a single fine-text OCR tier moving from 4 to 3 on one
// model in one think mode. It cost a day to localise from that symptom. Measured
// here at MLX ce916dbb: 232722 of 294912 output elements wrong by more than 1.0,
// max error 24.05. At d9add9d1 (post-fix): zero, max error 0.138.
//
// M MUST EXCEED get_qmv_batch_limit. quantized.cpp dispatches to the matrix-
// VECTOR kernel when M is at or below that limit, which for K=4304, N=1152 is
// 13. A first attempt at M=8 never reached fp_qmm_t, came back clean on the
// buggy library, and read as a refutation of a correct diagnosis. If this test
// is ever "simplified" by shrinking M, it stops testing anything.
func TestFpQmmTKMod32(t *testing.T) {
	withMLXThread(t, func(t *mlxthreadtest.T) {
		const (
			N     = 1152 // gemma4 vision tower down_proj output dim
			M     = 256  // must exceed get_qmv_batch_limit (13 here); see above
			group = 16
			bits  = 4
			// Post-fix the aligned K lands at 1.5e-4 and the trigger K at 0.14,
			// both pure quantisation error. The bug produced 24.05. Anything
			// above 1.0 is the kernel, not the quantiser.
			tolerance = 1.0
		)

		for _, tc := range []struct {
			name string
			K    int
		}{
			{"aligned_control", 4288},         // K mod 32 == 0
			{"gemma4_vision_down_proj", 4304}, // K mod 32 == 16, the trigger
		} {
			func() {
				rng := rand.New(rand.NewSource(42))
				wv := make([]float32, N*tc.K)
				for i := range wv {
					wv[i] = float32(rng.NormFloat64())
				}
				xv := make([]float32, M*tc.K)
				for i := range xv {
					xv[i] = float32(rng.NormFloat64())
				}

				w := FromValues(wv, N, tc.K)
				x := FromValues(xv, M, tc.K)
				wq, scales, biases := Quantize(w, group, bits, "nvfp4")

				// Reference deliberately avoids fp_qmm_t: dequantize, then a
				// dense matmul. Any disagreement beyond quantisation error is
				// the quantized kernel.
				deq := Dequantize(wq, scales, biases, group, bits, "nvfp4", nil)
				want := Matmul(x, Transpose(deq, 1, 0))
				got := QuantizedMatmul(x, wq, scales, biases, true, group, bits, "nvfp4", nil)
				Eval(want, got)

				a, b := want.Floats(), got.Floats()
				if len(a) != len(b) {
					t.Fatalf("shape mismatch: reference %d elements, kernel %d", len(a), len(b))
				}
				var maxAbs float64
				var bad int
				for i := range a {
					d := math.Abs(float64(a[i]) - float64(b[i]))
					if d > maxAbs {
						maxAbs = d
					}
					if d > tolerance {
						bad++
					}
				}
				t.Logf("%s: K=%d (K%%32=%d) max|qmm-reference|=%.6g, %d/%d elements over %.1f",
					tc.name, tc.K, tc.K%32, maxAbs, bad, len(a), tolerance)
				if bad > 0 {
					t.Errorf("fp_qmm_t disagrees with dequantize+matmul at K=%d (K%%32=%d): "+
						"%d of %d elements off by more than %.1f, max %.6g. "+
						"This is the MLX #3912 signature; check the MLX pin (%s).",
						tc.K, tc.K%32, bad, len(a), tolerance, maxAbs, Version())
				}
			}()
		}
	})
}
