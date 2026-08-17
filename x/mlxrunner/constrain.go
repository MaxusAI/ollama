package mlxrunner

import (
	"errors"
	"fmt"
	"math"
	"os"
	"strings"

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

// draftMasks returns the grammar mask for each verification row of a
// speculative step, the number of leading drafts the grammar admits, and a
// matcher advanced over exactly those drafts.
//
// STAGE 2 OF GRAMMAR-AWARE SPECULATION, host half. Verification compares the
// target model's distribution against the draft's, and under a grammar that
// comparison MUST use the MASKED target distribution. Otherwise the target's
// argmax can be a token the format forbids, and every legal draft is rejected
// against a token that could never have been emitted — speculation would
// "work" and accept nothing.
//
// ROW i PREDICTS DRAFT i, so row i carries the mask of the state BEFORE that
// draft: row 0 is the current state, row i is the state after drafts[0..i-1].
// The returned slice has legal+1 entries — one per admitted draft plus the
// bonus row, which is the state after the last admitted draft and is where the
// next token is sampled when every draft is accepted.
//
// The caller's matcher is not mutated, for the reason draftPrefix documents: a
// draft is a guess, and advancing the live matcher over a rejected guess would
// leave the request generating against a state its own output never reached.
//
// Building the device-side bias from these masks is the other half and is not
// here: it is a [legal+1, vocabDim] float32 upload, and its cost has to be
// measured against the 42% it is meant to recover rather than assumed.
func draftMasks(m *structured.Matcher, v *structured.Vocab, pieces [][]byte,
	isEOS func(int32) bool, drafts []int32,
) ([]*structured.Mask, int, *structured.Matcher) {
	cur := m.Clone()
	masks := make([]*structured.Mask, 0, len(drafts)+1)
	for i, id := range drafts {
		mask := v.Mask(cur)
		masks = append(masks, mask)
		if !mask.Allowed(id) {
			return masks, i, cur
		}
		if isEOS(id) {
			// EOS is admitted but nothing follows it, so there is no bonus row:
			// the run ends here and the caller samples nothing further.
			return masks, i + 1, cur
		}
		var piece []byte
		if int(id) >= 0 && int(id) < len(pieces) {
			piece = pieces[id]
		}
		if len(piece) == 0 || !cur.Advance(piece) {
			return masks, i, cur
		}
	}
	// Every draft admitted: the bonus row's mask is the state after the last.
	return append(masks, v.Mask(cur)), len(drafts), cur
}

// GrammarSpeculationEnv gates grammar-aware speculation. UNSET IS OFF, and off
// is the shipped behaviour: the mechanism below has never run against a real
// model, and it edits the one path where a mistake is silent KV corruption
// rather than a crash. Turn it on to measure it, not to use it.
const GrammarSpeculationEnv = "OLLAMA_MLX_GRAMMAR_SPECULATION"

func grammarSpeculationEnabled() bool {
	v := os.Getenv(GrammarSpeculationEnv)
	return v == "1" || strings.EqualFold(v, "true")
}

// fillMaskBias lays out the [len(masks), vocabDim] logit bias for masked
// verification: 0 where a row's mask admits the token, -Inf where it does not.
//
// Row-major and contiguous, because mlx.FromValues takes a flat slice and the
// shape is applied on top. buf is reused across steps -- this runs once per
// speculative round, and at k=3 on gemma4's 262,144 vocabulary it is 4 rows of
// 1 MiB, so reallocating it every round would be the dominant cost of the
// feature rather than an incidental one.
//
// Separated from the device upload so it can be tested without MLX: the shape
// and the -Inf placement are where an off-by-one would hide, and both are
// checkable on any host.
func fillMaskBias(masks []*structured.Mask, vocabDim int, buf []float32) []float32 {
	need := len(masks) * vocabDim
	if cap(buf) < need {
		buf = make([]float32, need)
	}
	buf = buf[:need]
	negInf := float32(math.Inf(-1))
	for i := range buf {
		buf[i] = negInf
	}
	for row, mask := range masks {
		base := row * vocabDim
		mask.ForEach(func(id int32) {
			if int(id) < vocabDim {
				buf[base+int(id)] = 0
			}
		})
	}
	return buf
}

// constraintBiasRows uploads the per-row mask bias for masked verification,
// shaped [rows, vocabDim] to line up with the fused hidden's rows.
//
// UNVERIFIED AGAINST HARDWARE. Everything below the fill is written from the
// shape contract of Sampler.Distribution and has never run against a real
// model; the gate exists so it cannot reach anyone who has not chosen to
// measure it. The fill itself is tested (constrain_bias_test.go) because that
// is where an off-by-one would hide and it is checkable without MLX.
func constraintBiasRows(masks []*structured.Mask, vocabDim int, buf []float32) (*mlx.Array, []float32) {
	buf = fillMaskBias(masks, vocabDim, buf)
	return mlx.FromValues(buf, len(masks), vocabDim), buf
}

// maskedTargetLogits biases a speculative round's logits so verification
// compares against the target's MASKED distribution.
//
// Without this, verification compares a draft against the target's UNMASKED
// argmax, which can be a token the format forbids: every legal draft is then
// rejected against a token that could never have been emitted, and speculation
// accepts nothing while appearing to work. That failure reads as "speculation
// did not help" rather than as a bug, which is why it is stated here.
//
// Returns the biased logits and the number of draft rows the grammar admits.
// The caller must verify only that many drafts: rows past the first illegal
// token describe grammar states that do not exist, so their masks would be
// fiction and any acceptance from them would be corruption.
func (s *speculationSession) maskedTargetLogits(logits *mlx.Array, draftIDs []int32) (*mlx.Array, int) {
	if s.matcher == nil {
		return logits, len(draftIDs)
	}
	masks, legal, _ := draftMasks(s.matcher, s.vocab, s.pieces, s.spec.r.Tokenizer.IsEOS, draftIDs)
	vocabDim := logits.Dim(logits.NumDims() - 1)
	bias, buf := constraintBiasRows(masks, vocabDim, s.biasBuf)
	s.biasBuf = buf
	// The forward produced len(draftIDs)+1 rows; the bias covers len(masks).
	// Only the leading rows are biased and only they may be verified.
	return mlx.Add(logits.Slice(mlx.Slice(), mlx.Slice(0, len(masks)), mlx.Slice()), bias), legal
}

// adoptGrammar advances the session's matcher over tokens verification
// committed. Called only after commit, never on a draft: a matcher advanced
// over tokens that were then rolled back would leave the request generating
// against a state its own output never reached.
func (s *speculationSession) adoptGrammar(ids []int32) {
	if s.matcher == nil {
		return
	}
	for _, id := range ids {
		if s.spec.r.Tokenizer.IsEOS(id) {
			return
		}
		if int(id) >= 0 && int(id) < len(s.pieces) && len(s.pieces[id]) > 0 {
			s.matcher.Advance(s.pieces[id])
		}
	}
}

// attachGrammar gives a speculation session the state masked verification
// needs. Called only on the gated path; without it s.matcher stays nil and
// every masked branch is inert.
func (s *speculationSession) attachGrammar(r *Runner, g *structured.Grammar) {
	vocab, pieces := r.constraint()
	s.matcher = g.NewMatcher()
	s.vocab = vocab
	s.pieces = pieces
}

// errNoLegalDraft ends a speculative round in which the grammar admitted none
// of the drafted tokens. Not a failure: the caller falls back to a serial step,
// which is what the unconstrained path would have produced anyway.
var errNoLegalDraft = errors.New("speculation: grammar admitted no drafted token")
