package mlxrunner

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/structured"
)

// constraintBias renders mask as an additive logit bias of shape
// [1, vocabDim]: zero for allowed ids, -inf for everything else including
// padded logit positions past the tokenizer vocabulary. buf is reused
// across steps to avoid a per-token allocation.
func constraintBias(mask *structured.Mask, vocabDim int, buf []float32) (*mlx.Array, []float32) {
	if cap(buf) < vocabDim {
		buf = make([]float32, vocabDim)
	}
	buf = buf[:vocabDim]
	negInf := float32(math.Inf(-1))
	for i := range buf {
		buf[i] = negInf
	}
	mask.ForEach(func(id int32) {
		if int(id) < vocabDim {
			buf[id] = 0
		}
	})
	return mlx.FromValues(buf, 1, vocabDim), buf
}

// constraint returns the lazily built vocabulary index for masking, plus
// the per-id decoded pieces used to advance the matcher over sampled
// tokens. Special tokens and EOS decode to sentinel text, so they are
// excluded from the pieces; EOS ids are legal exactly when the grammar
// can complete.
func (r *Runner) constraint() (*structured.Vocab, [][]byte) {
	r.constraintOnce.Do(func() {
		t := r.Tokenizer
		skip := make(map[int32]bool)
		for _, id := range t.SpecialTokenIDs() {
			skip[id] = true
		}
		for _, id := range t.EOSTokens() {
			skip[id] = true
		}
		pieces := make([][]byte, t.VocabSize())
		for id := range pieces {
			if skip[int32(id)] {
				continue
			}
			if s := t.Decode([]int32{int32(id)}); s != "" {
				pieces[id] = []byte(s)
			}
		}
		r.constraintPieces = pieces
		r.constraintVocab = structured.NewVocab(pieces, t.EOSTokens())
	})
	return r.constraintVocab, r.constraintPieces
}

// constrainedDecoder decodes one token per call with the format grammar's
// token mask applied before sampling. The next forward pass is dispatched
// before the previous token is read, so the grammar advance and mask
// computation on the CPU overlap the forward running on the GPU; only the
// sampling op waits for the mask.
type constrainedDecoder struct {
	r    *Runner
	spec *speculationSession // kept in lockstep; never proposes

	caches   []cache.Cache
	position int

	matcher *structured.Matcher
	vocab   *structured.Vocab
	pieces  [][]byte
	biasBuf []float32

	pending sampler.Result // sampled, not yet forwarded
}

func (r *Runner) constrainedDecoder(spec *speculationSession, caches []cache.Cache, seed *mlx.Array, position int, grammar *structured.Grammar) *constrainedDecoder {
	vocab, pieces := r.constraint()
	d := &constrainedDecoder{
		r:        r,
		spec:     spec,
		caches:   caches,
		position: position,
		matcher:  grammar.NewMatcher(),
		vocab:    vocab,
		pieces:   pieces,
	}
	d.pending = d.sampleMasked(d.forward(seed))
	return d
}

// forward runs one forward pass, keeping any drafter's KV in lockstep,
// and returns the last-position logits ([1, V], lazy).
func (d *constrainedDecoder) forward(token *mlx.Array) *mlx.Array {
	r := d.r
	hidden, auxHidden := r.Model.Forward(&batch.Batch{
		InputIDs:     token,
		SeqOffsets:   []int32{int32(d.position)},
		SeqQueryLens: []int32{int32(token.Dim(1))},
	}, d.caches)
	// auxHidden is the draft-conditioning state upstream's Forward now returns;
	// decode forwards carry no media, hence the nil.
	d.spec.committed(token, auxHidden, d.position, nil)
	d.position += token.Dim(1)
	logits := r.Model.Unembed(hidden)
	return logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)
}

// sampleMasked applies the current grammar state's token mask to logits
// and samples one token, leaving its evaluation in flight.
func (d *constrainedDecoder) sampleMasked(logits *mlx.Array) sampler.Result {
	mask := d.vocab.Mask(d.matcher)
	var bias *mlx.Array
	bias, d.biasBuf = constraintBias(mask, logits.Dim(logits.NumDims()-1), d.biasBuf)
	next := d.r.Sampler.Sample([]int{pipelineSlot}, mlx.Add(logits, bias))
	mlx.Pin(next.Arrays()...)
	mlx.Sweep()
	mlx.AsyncEval(next.Arrays()...)
	return next
}

func (d *constrainedDecoder) next(int) ([]sampler.Result, error) {
	out := d.pending

	// Dispatch the forward before reading the token so the GPU runs
	// while the matcher advances and the next mask is built.
	logits := d.forward(out.Token.ExpandDims(-1))

	id := int32(out.Token.Int())
	if !d.r.Tokenizer.IsEOS(id) {
		var piece []byte
		if int(id) < len(d.pieces) {
			piece = d.pieces[id]
		}
		if len(piece) == 0 || !d.matcher.Advance(piece) {
			// The mask guarantees legality; reaching this is a bug, and
			// generating past it would break the format promise.
			return nil, fmt.Errorf("constrained sampling produced an illegal token %d (%q)", id, piece)
		}
	}

	d.pending = d.sampleMasked(logits)
	mlx.Unpin(out.Arrays()...)
	return []sampler.Result{out}, nil
}

// drain ends production: the in-flight sample was never forwarded; the
// decoder keeps it for close.
func (d *constrainedDecoder) drain() ([]sampler.Result, int) {
	return []sampler.Result{d.pending}, d.position
}

func (d *constrainedDecoder) close() {
	d.spec.settle(d.pending.Token)
	mlx.Unpin(d.pending.Arrays()...)
}

// draftPrefix reports how many leading draft tokens the grammar admits, and
// returns a matcher advanced over exactly those tokens.
//
// STAGE 1 OF GRAMMAR-AWARE SPECULATION. Constrained decoding currently runs one
// token per forward pass (constrainedDecoder.next), while unconstrained
// decoding is speculative. Measured on CUDA, gemma4:31b-nvfp4, one 1920x1080
// image, n=3: 21.13 tok/s with format:"json" against 36.53 without, a 42%
// penalty, where llama-server pays 1.6% for the same constraint because it
// verifies drafts against the grammar instead of giving up on drafting. See
// ../mlx-constrained-decode-disables-speculation.md.
//
// This is the host half of closing that gap: given a draft, decide how much of
// it the grammar could possibly accept, so verification never has to consider
// a token the format forbids. It does NOT decide acceptance -- the target
// model's masked distribution does that, and it is the device half, not yet
// written. A token counted here is a token still eligible to be accepted.
//
// THE CALLER'S MATCHER IS NOT MUTATED. Grammar state is a stack machine and a
// draft is a guess; advancing the live matcher over a guess that verification
// then rejects would leave the request generating against a state its own
// output never reached. So this walks a Clone and hands it back: adopt it only
// for the tokens verification actually commits, which is why the count and the
// matcher are returned together rather than the matcher being advanced in
// place.
//
// EOS ends the walk rather than extending it. Vocab.Mask only admits EOS where
// the grammar CanComplete, so an EOS reaching this point is legal by
// construction; but nothing follows it, and advancing past it would ask the
// matcher to consume a piece that decodes to sentinel text.
func draftPrefix(m *structured.Matcher, v *structured.Vocab, pieces [][]byte,
	isEOS func(int32) bool, drafts []int32,
) (int, *structured.Matcher) {
	cur := m.Clone()
	for i, id := range drafts {
		if !v.Mask(cur).Allowed(id) {
			return i, cur
		}
		if isEOS(id) {
			return i + 1, cur
		}
		var piece []byte
		if int(id) >= 0 && int(id) < len(pieces) {
			piece = pieces[id]
		}
		// A zero-length piece is a special or undecodable token. The serial
		// path treats reaching one as a bug because the mask guarantees
		// legality; here it is merely the end of what can be verified, since
		// the matcher cannot be advanced over text that does not exist.
		if len(piece) == 0 || !cur.Advance(piece) {
			return i, cur
		}
	}
	return len(drafts), cur
}
