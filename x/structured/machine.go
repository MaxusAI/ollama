// Package structured compiles /api request format values ("json" or a JSON
// Schema object) into byte-level grammars and computes per-step token masks
// for constrained sampling. Pure Go: no MLX, no cgo.
//
// The matcher uses the same formalization as llama.cpp's llama-grammar: a
// set of pushdown stacks advanced in lockstep, forking on alternation. It
// operates on bytes rather than codepoints; JSON's structural characters are
// ASCII and string contents admit all bytes >= 0x20 except '"' and '\\', so
// the only relaxation is that multi-byte UTF-8 well-formedness is not
// enforced by the grammar itself.
package structured

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"slices"
	"sort"
	"sync/atomic"
)

// byteSet is a 256-bit set of byte values.
type byteSet [4]uint64

func (s *byteSet) add(b byte)      { s[b>>6] |= 1 << (b & 63) }
func (s *byteSet) has(b byte) bool { return s[b>>6]&(1<<(b&63)) != 0 }

func (s *byteSet) addRange(lo, hi byte) {
	for b := int(lo); b <= int(hi); b++ {
		s.add(byte(b))
	}
}

func (s *byteSet) invert() {
	for i := range s {
		s[i] = ^s[i]
	}
}

type elemKind uint8

const (
	elemBytes elemKind = iota // match one byte from set
	elemRef                   // descend into rule ref
)

type elem struct {
	kind elemKind
	set  byteSet
	ref  int32
}

// seq is one alternative of a rule: a sequence of elements. An empty seq is
// epsilon.
type seq []elem

type rule struct {
	alts []seq
}

var grammarIDs atomic.Uint64

// Grammar is a compiled format constraint.
type Grammar struct {
	rules []rule
	root  int32
	id    uint64
}

// ID uniquely identifies this compiled grammar within the process, so mask
// caches can key on (grammar, state).
func (g *Grammar) ID() uint64 { return g.id }

// Compile parses a format value: the JSON string "json" or a JSON Schema
// object. Anything else — including empty and null, which callers are
// expected to treat as "no format" without calling Compile — is an error.
func Compile(format []byte) (*Grammar, error) {
	trimmed := bytes.TrimSpace(format)
	switch {
	case string(trimmed) == `"json"`:
		g := jsonGrammar()
		g.id = grammarIDs.Add(1)
		return g, nil
	case len(trimmed) > 0 && trimmed[0] == '{':
		g, err := schemaGrammar(trimmed)
		if err != nil {
			return nil, err
		}
		g.id = grammarIDs.Add(1)
		return g, nil
	default:
		display := format
		if len(display) > 64 {
			display = display[:64]
		}
		return nil, fmt.Errorf(`invalid format %q: expected "json" or a JSON Schema object`, display)
	}
}

// frame is one immutable stack frame: a position (rule, alt, idx) plus the
// parent it returns to. Sharing parents makes forking a stack O(1).
type frame struct {
	parent *frame
	hash   uint64
	rule   int32
	alt    int32
	idx    int32
}

func frameValue(parent *frame, ruleIdx, alt, idx int32) frame {
	var ph uint64
	if parent != nil {
		ph = parent.hash
	}
	return frame{
		parent: parent,
		hash:   mix64(ph ^ uint64(ruleIdx)<<42 ^ uint64(alt)<<21 ^ uint64(idx) ^ 0x9e3779b97f4a7c15),
		rule:   ruleIdx,
		alt:    alt,
		idx:    idx,
	}
}

func newFrame(parent *frame, ruleIdx, alt, idx int32) *frame {
	f := frameValue(parent, ruleIdx, alt, idx)
	return &f
}

func mix64(z uint64) uint64 {
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

func frameEqual(a, b *frame) bool {
	for a != nil && b != nil {
		if a == b {
			return true
		}
		if a.rule != b.rule || a.alt != b.alt || a.idx != b.idx {
			return false
		}
		a, b = a.parent, b.parent
	}
	return a == nil && b == nil
}

// completeHash keys the empty ("input is a complete match") stack in state
// sets; no frame chain hashes to it in practice.
const completeHash = 0x517cc1b727220a95

func stackHash(f *frame) uint64 {
	if f == nil {
		return completeHash
	}
	return f.hash
}

// Matcher is a mutable cursor over a Grammar, advanced byte by byte. A
// state is a set of stacks; the top of every stack is a byte element
// (normalization guarantees it), and a nil stack means the input so far is
// a complete match.
type Matcher struct {
	g      *Grammar
	stacks []*frame
}

// NewMatcher returns a matcher positioned at the grammar's start state.
func (g *Grammar) NewMatcher() *Matcher {
	m := &Matcher{g: g}
	n := normalizer{g: g, alloc: heapFrame}
	for alt := range g.rules[g.root].alts {
		n.normalize(nil, g.root, int32(alt), 0)
	}
	m.stacks = n.out
	return m
}

// Clone returns an independent matcher sharing the (immutable) frames.
func (m *Matcher) Clone() *Matcher {
	return &Matcher{g: m.g, stacks: slices.Clone(m.stacks)}
}

func heapFrame(parent *frame, ruleIdx, alt, idx int32) *frame {
	return newFrame(parent, ruleIdx, alt, idx)
}

// normalizer accumulates stable stacks — stacks whose top element is a byte
// matcher (or which are complete) — deduplicating along the way. State
// sets stay tiny for JSON-shaped grammars, so dedup is a linear scan; the
// alloc hook lets mask computation use a slab instead of the heap.
type normalizer struct {
	g     *Grammar
	out   []*frame
	alloc func(parent *frame, ruleIdx, alt, idx int32) *frame

	steps int                  // ε-expansion steps in the current normalize
	seen  map[normPos]struct{} // positions already expanded, once over budget
}

// normPos identifies an ε-expansion position. Expanding one is a pure
// function of the position, so a position reached twice can be skipped.
type normPos struct {
	parent *frame
	rule   int32
	alt    int32
	idx    int32
}

func (n *normalizer) add(f *frame) {
	h := stackHash(f)
	for _, prev := range n.out {
		if stackHash(prev) == h && frameEqual(prev, f) {
			return
		}
	}
	n.out = append(n.out, f)
}

// normalizeDepthLimit bounds the ε-expansion recursion so degenerate
// grammars (mutually recursive unit rules a ::= b; b ::= a) cannot hang a
// request; expansion paths past the limit are dropped, which can only
// shrink the accepted language of an already-degenerate grammar.
const normalizeDepthLimit = 4096

// normalizeWorkBudget is how many ε-expansion steps one normalize may take
// before it starts remembering the positions it has expanded. The depth
// limit bounds how deep expansion goes but not how wide: a grammar whose
// cycle is made of tail references allocates no frame and never reaches
// n.add, so its child-alternative loop explores one path per combination —
// exponentially many. Memoizing collapses those paths to the (finite) set
// of distinct positions without changing what is expanded, and staying off
// the map until the budget is spent keeps the per-token mask walk, where
// real grammars finish in a few dozen steps, allocation-free.
const normalizeWorkBudget = 1024

// normalize expands the position (parent, rule, alt, idx) until every
// resulting stack is stable: past-the-end positions pop to the parent,
// rule references descend into each alternative. Frames are only
// materialized for stable stacks. A reference in tail position reuses the
// parent continuation instead of pushing a frame — without this, star
// recursions like string content grow the stack by one frame per consumed
// byte, so states never repeat, mask caching never hits, and long
// generations slow down quadratically.
func (n *normalizer) normalize(parent *frame, ruleIdx, alt, idx int32) {
	n.steps = 0
	n.normalizeAt(parent, ruleIdx, alt, idx, 0)
}

func (n *normalizer) normalizeAt(parent *frame, ruleIdx, alt, idx int32, depth int) {
	if depth > normalizeDepthLimit {
		return
	}
	n.steps++
	if n.steps > normalizeWorkBudget {
		p := normPos{parent: parent, rule: ruleIdx, alt: alt, idx: idx}
		if _, done := n.seen[p]; done {
			return
		}
		if n.seen == nil {
			n.seen = make(map[normPos]struct{})
		}
		n.seen[p] = struct{}{}
	}
	s := n.g.rules[ruleIdx].alts[alt]
	if int(idx) >= len(s) {
		// Sequence exhausted: return to the parent, past the ref that
		// brought us here.
		if parent == nil {
			n.add(nil)
			return
		}
		n.normalizeAt(parent.parent, parent.rule, parent.alt, parent.idx+1, depth+1)
		return
	}
	e := &s[idx]
	if e.kind == elemBytes {
		n.add(n.alloc(parent, ruleIdx, alt, idx))
		return
	}
	next := parent
	if int(idx)+1 < len(s) {
		// Only a non-tail reference needs a return frame; a tail call
		// continues straight into the parent's continuation.
		next = n.alloc(parent, ruleIdx, alt, idx)
	}
	for childAlt := range n.g.rules[e.ref].alts {
		n.normalizeAt(next, e.ref, int32(childAlt), 0, depth+1)
	}
}

// advanceStacks advances every stack in stacks over byte b, appending the
// resulting stable stacks to n.out. It reports whether any survived.
func (n *normalizer) advanceStacks(stacks []*frame, b byte) bool {
	for _, st := range stacks {
		if st == nil {
			// A complete stack consumes nothing.
			continue
		}
		e := &n.g.rules[st.rule].alts[st.alt][st.idx]
		if e.set.has(b) {
			n.normalize(st.parent, st.rule, st.alt, st.idx+1)
		}
	}
	return len(n.out) > 0
}

// AdvanceByte consumes b. It returns false — leaving the state unchanged —
// when b is not a legal continuation.
func (m *Matcher) AdvanceByte(b byte) bool {
	n := normalizer{g: m.g, alloc: heapFrame}
	if !n.advanceStacks(m.stacks, b) {
		return false
	}
	m.stacks = n.out
	return true
}

// Advance consumes bs atomically: on any illegal byte it returns false and
// the state is unchanged.
func (m *Matcher) Advance(bs []byte) bool {
	saved := m.stacks
	for _, b := range bs {
		if !m.AdvanceByte(b) {
			m.stacks = saved
			return false
		}
	}
	return true
}

// CanComplete reports whether the input consumed so far is a complete match.
func (m *Matcher) CanComplete() bool {
	for _, st := range m.stacks {
		if st == nil {
			return true
		}
	}
	return false
}

// StateKey returns a hash identifying the current state. Equal states hash
// equal; distinct states collide only with negligible odds.
func (m *Matcher) StateKey() uint64 {
	hs := make([]uint64, len(m.stacks))
	for i, st := range m.stacks {
		hs[i] = stackHash(st)
	}
	sort.Slice(hs, func(i, j int) bool { return hs[i] < hs[j] })
	key := uint64(len(hs))
	for _, h := range hs {
		key = mix64(key ^ h)
	}
	return key
}

// builder assembles a Grammar. Rules may be reserved first and defined
// later so they can reference each other recursively.
type builder struct {
	g *Grammar
}

func newBuilder() *builder { return &builder{g: &Grammar{}} }

func (b *builder) reserve() int32 {
	b.g.rules = append(b.g.rules, rule{})
	return int32(len(b.g.rules) - 1)
}

func (b *builder) define(id int32, alts ...seq) int32 {
	b.g.rules[id] = rule{alts: alts}
	return id
}

func (b *builder) addRule(alts ...seq) int32 {
	return b.define(b.reserve(), alts...)
}

// group wraps alternatives in an anonymous rule and returns a ref element.
func (b *builder) group(alts ...seq) elem {
	return ref(b.addRule(alts...))
}

// opt makes (x)? for a sequence.
func (b *builder) opt(x seq) elem {
	return b.group(x, seq{})
}

// star makes (x)* via r ::= x r | epsilon.
func (b *builder) star(x seq) elem {
	id := b.reserve()
	b.define(id, append(slices.Clone(x), ref(id)), seq{})
	return ref(id)
}

// plus makes (x)+.
func (b *builder) plus(x seq) elem {
	return b.group(append(slices.Clone(x), b.star(x)))
}

// repeat makes x{min,max}; max < 0 means unbounded.
func (b *builder) repeat(x seq, minCount, maxCount int) elem {
	out := seq{}
	for range minCount {
		out = append(out, x...)
	}
	switch {
	case maxCount < 0:
		out = append(out, b.star(x))
	case maxCount < minCount:
		panic(fmt.Sprintf("structured: repeat max %d < min %d", maxCount, minCount))
	default:
		// Nest the optional tail: (x (x ...)?)?
		var tail elem
		haveTail := false
		for range maxCount - minCount {
			ext := slices.Clone(x)
			if haveTail {
				ext = append(ext, tail)
			}
			tail = b.opt(ext)
			haveTail = true
		}
		if haveTail {
			out = append(out, tail)
		}
	}
	return b.group(out)
}

// canonicalizePasses bounds the merge sweeps. A pass that merges nothing
// stops the loop, and stopping early only leaves rules unmerged, so the cap
// can slow convergence but never changes the accepted language.
const canonicalizePasses = 32

// canonicalize merges rules whose bodies agree element for element and then
// drops duplicate alternatives. Converters name every subschema separately,
// so the two branches of {"anyOf":[T,T]} compile to distinct rules with the
// same shape; if both stay live, each level of a recursion through them
// doubles the stack set, and Vocab.Mask pays that for every token. Merging
// is language-preserving: two rules are only merged once their alternatives
// match under the merges already made, and a reference back to the rule
// being keyed is encoded as "self", so the star rules that repetitions
// expand into (r ::= x r | ε) still compare equal to their twins.
func (g *Grammar) canonicalize() {
	rep := make([]int32, len(g.rules))
	for i := range rep {
		rep[i] = int32(i)
	}
	find := func(i int32) int32 {
		for rep[i] != i {
			rep[i] = rep[rep[i]]
			i = rep[i]
		}
		return i
	}

	var buf []byte
	for range canonicalizePasses {
		byBody := make(map[string]int32, len(g.rules))
		merged := false
		for i := range g.rules {
			if find(int32(i)) != int32(i) {
				continue
			}
			buf = appendRuleKey(buf[:0], &g.rules[i], find, int32(i))
			if j, dup := byBody[string(buf)]; dup {
				rep[i] = j
				merged = true
				continue
			}
			byBody[string(buf)] = int32(i)
		}
		if !merged {
			break
		}
	}

	for ri := range g.rules {
		r := &g.rules[ri]
		for _, alt := range r.alts {
			for k := range alt {
				if alt[k].kind == elemRef {
					alt[k].ref = find(alt[k].ref)
				}
			}
		}
		if len(r.alts) < 2 {
			continue
		}
		seen := make(map[string]bool, len(r.alts))
		kept := r.alts[:0]
		for _, alt := range r.alts {
			buf = appendSeqKey(buf[:0], alt, find, -1)
			if seen[string(buf)] {
				continue
			}
			seen[string(buf)] = true
			kept = append(kept, alt)
		}
		r.alts = kept
	}
	g.root = find(g.root)
}

// appendRuleKey appends a self-delimiting encoding of r's alternatives, so
// rules compare by value as map keys. self is the rule's own representative
// (-1 for none); references to it encode as "self" rather than as an id.
func appendRuleKey(buf []byte, r *rule, find func(int32) int32, self int32) []byte {
	for _, alt := range r.alts {
		buf = append(buf, '|')
		buf = appendSeqKey(buf, alt, find, self)
	}
	return buf
}

func appendSeqKey(buf []byte, s seq, find func(int32) int32, self int32) []byte {
	for _, e := range s {
		if e.kind == elemBytes {
			buf = append(buf, 'b')
			for _, w := range e.set {
				buf = binary.LittleEndian.AppendUint64(buf, w)
			}
			continue
		}
		if target := find(e.ref); target == self {
			buf = append(buf, 's')
		} else {
			buf = append(buf, 'r')
			buf = binary.LittleEndian.AppendUint32(buf, uint32(target))
		}
	}
	return buf
}

func ref(id int32) elem { return elem{kind: elemRef, ref: id} }

// cls builds a byte element from inclusive ranges, e.g. cls('a','z','0','9').
func cls(ranges ...byte) elem {
	if len(ranges)%2 != 0 {
		panic("structured: cls wants (lo,hi) pairs")
	}
	e := elem{kind: elemBytes}
	for i := 0; i < len(ranges); i += 2 {
		e.set.addRange(ranges[i], ranges[i+1])
	}
	return e
}

// chars builds a byte element from individual byte values.
func chars(bs ...byte) elem {
	e := elem{kind: elemBytes}
	for _, b := range bs {
		e.set.add(b)
	}
	return e
}

// lit builds a sequence matching s exactly.
func lit(s string) seq {
	out := make(seq, len(s))
	for i := range len(s) {
		out[i] = chars(s[i])
	}
	return out
}

// notChars builds a byte element matching every byte except the given ones
// and the given (lo,hi) ranges in excludeRanges.
func notChars(exclude []byte, excludeRanges ...byte) elem {
	e := elem{kind: elemBytes}
	for _, b := range exclude {
		e.set.add(b)
	}
	for i := 0; i < len(excludeRanges); i += 2 {
		e.set.addRange(excludeRanges[i], excludeRanges[i+1])
	}
	e.set.invert()
	return e
}

// jsonGrammar mirrors grammarJSON in llm/llama_server.go byte for byte:
//
//	root   ::= object
//	value  ::= object | array | string | number | ("true" | "false" | "null") ws
//	object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? ws "}"
//	array  ::= "[" ws ( value ("," ws value)* )? ws "]"
//	string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}) )* "\""
//	number ::= ("-"? ([0-9] | [1-9][0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
//	ws     ::= ([ \t\n] ws)?
func jsonGrammar() *Grammar {
	b := newBuilder()

	root := b.reserve()
	b.g.root = root
	object := b.reserve()
	array := b.reserve()
	value := b.reserve()
	str := b.reserve()
	number := b.reserve()
	ws := b.reserve()

	b.define(root, seq{ref(object)})

	b.define(ws, seq{cls(' ', ' ', '\t', '\t', '\n', '\n'), ref(ws)}, seq{})

	b.define(value,
		seq{ref(object)},
		seq{ref(array)},
		seq{ref(str)},
		seq{ref(number)},
		append(seq{b.group(lit("true"), lit("false"), lit("null"))}, ref(ws)),
	)

	member := seq{ref(str), chars(':'), ref(ws), ref(value)}
	b.define(object, seq{
		chars('{'), ref(ws),
		b.opt(append(slices.Clone(member), b.star(append(seq{chars(','), ref(ws)}, member...)))),
		ref(ws), chars('}'),
	})

	b.define(array, seq{
		chars('['), ref(ws),
		b.opt(append(seq{ref(value)}, b.star(seq{chars(','), ref(ws), ref(value)}))),
		ref(ws), chars(']'),
	})

	hex := cls('0', '9', 'a', 'f', 'A', 'F')
	strChar := b.group(
		seq{notChars([]byte{'"', '\\', 0x7F}, 0x00, 0x1F)},
		seq{chars('\\'), b.group(
			seq{chars('"', '\\', '/', 'b', 'f', 'n', 'r', 't')},
			seq{chars('u'), hex, hex, hex, hex},
		)},
	)
	b.define(str, seq{chars('"'), b.star(seq{strChar}), chars('"')})

	digit := cls('0', '9')
	b.define(number, seq{
		b.opt(lit("-")),
		b.group(seq{digit}, seq{cls('1', '9'), b.star(seq{digit})}),
		b.opt(seq{chars('.'), b.plus(seq{digit})}),
		b.opt(seq{chars('e', 'E'), b.opt(seq{chars('-', '+')}), b.plus(seq{digit})}),
	})

	b.g.canonicalize()
	return b.g
}
