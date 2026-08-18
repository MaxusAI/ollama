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
	layout   []any

	matcher *structured.Matcher
	vocab   *structured.Vocab
	pieces  [][]byte
	biasBuf []float32

	pending sampler.Result // sampled, not yet forwarded
}

func (r *Runner) constrainedDecoder(spec *speculationSession, caches []cache.Cache, seed *mlx.Array, position int, grammar *structured.Grammar) *constrainedDecoder {
	vocab, pieces := r.constraint()
	return r.constrainedStep(spec, caches, seed, position, nil, grammar.NewMatcher(), vocab, pieces)
}

// constrainedStep builds a constrained decoder over a matcher the caller owns.
//
// The serial path passes a fresh matcher and no layout. Grammar-aware
// speculation passes the SESSION'S matcher and the session's media layout, so
// that a parked round -- one where the engine cannot draft -- is still masked
// and still advances the one grammar state the request has. Sharing the
// matcher rather than cloning it is the point: park and draft alternate within
// a single request, and two matchers would drift the moment they did.
func (r *Runner) constrainedStep(spec *speculationSession, caches []cache.Cache, seed *mlx.Array, position int, layout []any, matcher *structured.Matcher, vocab *structured.Vocab, pieces [][]byte) *constrainedDecoder {
	d := &constrainedDecoder{
		r:        r,
		spec:     spec,
		caches:   caches,
		position: position,
		layout:   layout,
		matcher:  matcher,
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
		Layout:       d.layout,
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

// grammarRows plans a speculative round's masked verification: the per-row
// masks and the number of leading drafts the grammar admits.
//
// Row i predicts draft i, so mask i is the state BEFORE draft i. There are
// admitted+1 masks when a bonus row exists -- the state after the last
// admitted draft, where the next token is sampled if every draft is accepted
// -- and only admitted masks when the last admitted draft is EOS, because
// nothing follows an EOS.
//
// len(masks) == admitted IS THEREFORE THE CALLER'S SIGNAL that this round has
// no bonus row. accept handles it by dropping the drafted EOS, which restores
// the bonus row rather than carrying a round whose last row does not exist;
// see the comment there for why that is cheaper than the alternative.
//
// Returns nil masks when no grammar is attached, which is how every masked
// branch in accept stays inert on the unconstrained path.
func (s *speculationSession) grammarRows(draftIDs []int32) ([]*structured.Mask, int) {
	if s.matcher == nil {
		return nil, len(draftIDs)
	}
	masks, admitted, _ := draftMasks(s.matcher, s.vocab, s.pieces, s.isEOS, draftIDs)
	return masks, admitted
}

// verifyPlan decides how much of a draft this round verifies, and returns the
// masks that bias those rows.
//
// GUARANTEE: whenever verify > 0, len(masks) == verify+1 -- one mask per
// verified draft plus the bonus row. accept depends on that exactly, and it is
// the reason the drafted EOS is dropped here rather than special-cased there.
//
// An admitted EOS ends grammarRows' mask sequence a row early, because nothing
// follows an EOS and there is no state to sample a bonus token from. Carrying
// such a round would index sampleTokenAt past the distribution AND hand
// Sampler.Distribution as many rows as draft tokens -- which that function
// reads as a proposal-shaped call and silently shifts every row's repeat
// penalty history by one. Dropping the EOS instead costs one unaccepted draft
// per request and nothing else: the row before the EOS still admits EOS (the
// grammar can complete there, which is why the draft was legal at all), so the
// target can still sample it as the bonus token.
func (s *speculationSession) verifyPlan(draftIDs []int32) ([]*structured.Mask, int) {
	masks, admitted := s.grammarRows(draftIDs)
	// The guard on len(masks) matters: without a matcher grammarRows returns no
	// masks and admitted == len(draftIDs), and 0 == 0 would drive verify to -1.
	if len(masks) > 0 && len(masks) == admitted {
		admitted--
	}
	return masks, admitted
}

// maskRows biases a speculative round's logits so verification compares
// against the target's MASKED distribution.
//
// Without this, verification compares a draft against the target's UNMASKED
// argmax, which can be a token the format forbids: every legal draft is then
// rejected against a token that could never have been emitted, and speculation
// accepts nothing while appearing to work. That failure reads as "speculation
// did not help" rather than as a bug, which is why it is stated here.
//
// The bias covers len(masks) rows and the forward may have produced more, so
// only the leading len(masks) rows survive -- and only they may be verified.
// Rows past the last mask describe grammar states that do not exist, so their
// masks would be fiction and any acceptance from them would be corruption.
func (s *speculationSession) maskRows(logits *mlx.Array, masks []*structured.Mask) *mlx.Array {
	if len(masks) == 0 {
		return logits
	}
	bias, buf := constraintBiasRows(masks, logits.Dim(logits.NumDims()-1), s.biasBuf)
	s.biasBuf = buf
	return mlx.Add(logits.Slice(mlx.Slice(), mlx.Slice(0, len(masks)), mlx.Slice()), bias)
}

// adoptGrammar advances the session's matcher over tokens the request has
// EMITTED. Never call it on a draft: a matcher advanced over tokens that were
// then rolled back would leave the request generating against a state its own
// output never reached.
//
// THE INVARIANT IT MAINTAINS: the matcher describes every token emitted so
// far, INCLUDING the one held as current. accept relies on it -- mask row 0 is
// the state after current -- and it is what makes the mask sequence describe
// the text actually generated rather than a prefix of it. Emitting a token
// without adopting it silently shifts every later mask by one position, which
// then rejects legal drafts, which parks the round, which emits more
// unadopted tokens. The drift does not converge; it compounds.
//
// Every emission path must therefore reach this or advance the matcher itself:
// accept adopts its committed drafts AND the residual or bonus token it
// returns, resume adopts the token it drains, and a parked round advances the
// shared matcher inside constrainedDecoder.next.
//
// AN ILLEGAL TOKEN IS AN ERROR, not something to walk past. Every token
// reaching here was drawn from a masked distribution, so Advance can only fail
// if the matcher has drifted from the emitted text -- and generating past that
// point ships malformed output under a format promise. The serial decoder has
// made the same call since it was written (see constrainedDecoder.next); the
// gated path silently discarding the failure is how the drift below went
// unnoticed for a whole measurement campaign.
func (s *speculationSession) adoptGrammar(ids []int32) error {
	if s.matcher == nil {
		return nil
	}
	for _, id := range ids {
		if s.isEOS(id) {
			return nil
		}
		var piece []byte
		if int(id) >= 0 && int(id) < len(s.pieces) {
			piece = s.pieces[id]
		}
		if len(piece) == 0 || !s.matcher.Advance(piece) {
			return fmt.Errorf("grammar-aware speculation emitted an illegal token %d (%q)", id, piece)
		}
	}
	return nil
}

// attachGrammar gives a speculation session the state masked verification
// needs. Called only on the gated path; without it s.matcher stays nil and
// every masked branch is inert.
func (s *speculationSession) attachGrammar(r *Runner, g *structured.Grammar) {
	vocab, pieces := r.constraint()
	s.matcher = g.NewMatcher()
	s.vocab = vocab
	s.pieces = pieces
	s.isEOS = r.Tokenizer.IsEOS
}

// errNoLegalDraft ends a speculative round in which the grammar admitted none
// of the drafted tokens. Not a failure: the caller falls back to a serial step,
// which is what the unconstrained path would have produced anyway.
var errNoLegalDraft = errors.New("speculation: grammar admitted no drafted token")
