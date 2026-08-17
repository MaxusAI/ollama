package mlxrunner

import (
	"math/rand"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxtest"
	"github.com/ollama/ollama/x/structured"
)

// benchVocab mirrors x/structured's own benchmark vocabulary: every single
// byte plus ~262k word-like pieces, sized to gemma4's real vocabulary
// (embed_tokens is [262144, 672]). The size is the point — the per-token cost
// under test is linear in it.
func benchVocab(size int) (*structured.Vocab, int) {
	rng := rand.New(rand.NewSource(1))
	pieces := make([][]byte, 0, size)
	for b := range 256 {
		pieces = append(pieces, []byte{byte(b)})
	}
	const letters = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.,:\"{}[]"
	seen := map[string]bool{}
	for len(pieces) < size {
		n := 1 + rng.Intn(12)
		buf := make([]byte, n)
		for i := range buf {
			buf[i] = letters[rng.Intn(len(letters))]
		}
		if seen[string(buf)] {
			continue
		}
		seen[string(buf)] = true
		pieces = append(pieces, buf)
	}
	eos := int32(len(pieces))
	pieces = append(pieces, nil)
	return structured.NewVocab(pieces, []int32{eos}), len(pieces)
}

func benchMask(tb testing.TB, v *structured.Vocab) *structured.Mask {
	tb.Helper()
	g, err := structured.Compile([]byte(`"json"`))
	if err != nil {
		tb.Fatal(err)
	}
	m := g.NewMatcher()
	// In-string is the widest allowed set and the state a JSON generation
	// spends most of its tokens in, so it is the honest case to measure.
	m.Advance([]byte(`{"key`))
	return v.Mask(m)
}

// BenchmarkConstraintBias measures the per-decode-token cost of turning a
// grammar mask into logit bias.
//
// WHY THIS EXISTS. x/structured's mask computation was already benchmarked
// (BenchmarkMaskWarm, BenchmarkMaskGenerationSequence) and is cached on the
// grammar state key. The step AFTER it was not benchmarked at all, and it is
// not cached: constraintBias fills a vocabDim float32 buffer with -Inf, walks
// the mask, and uploads a fresh device array — every token, including when the
// mask is a cache hit and the bias is therefore provably identical to the
// previous step's.
//
// Measured on CUDA, gemma4:31b-nvfp4, one 1920x1080 image, n=3: decode runs at
// 21.1 tok/s with format:"json" against 36.5 unconstrained, a 42% penalty. The
// same comparison on llama-server is 57.2 against 58.2 — 1.6%. So the cost is
// ours, not a property of constrained decoding, and this is the loop it lives
// in.
func BenchmarkConstraintBias(b *testing.B) {
	mlxtest.Setup(b)
	v, vocabDim := benchVocab(262144)
	mask := benchMask(b, v)
	var buf []float32

	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		_, buf = constraintBias(mask, vocabDim, buf)
	}
}

// BenchmarkConstraintBiasFill isolates the CPU half — the -Inf memset and the
// mask walk — from the device upload, so a change can be attributed to one or
// the other. Runs without MLX.
func BenchmarkConstraintBiasFill(b *testing.B) {
	v, vocabDim := benchVocab(262144)
	mask := benchMask(b, v)
	buf := make([]float32, vocabDim)
	negInf := float32(-1)

	b.ResetTimer()
	b.ReportAllocs()
	for range b.N {
		for i := range buf {
			buf[i] = negInf
		}
		mask.ForEach(func(id int32) {
			if int(id) < vocabDim {
				buf[id] = 0
			}
		})
	}
}
