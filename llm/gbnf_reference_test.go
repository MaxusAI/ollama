package llm

import (
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"unicode/utf8"
)

// gbnfRecognizer is a reference recognizer for the GBNF subset the fork's
// own grammars use (rules, literals, character classes with ranges and
// negation, groups, alternation, ?, *, + and {m,n}), so grammarJSON's
// language can be checked in plain Go wherever the tests run. It parses as
// llama.cpp's grammar parser does (a rule ends at a newline outside
// parentheses; # comments run to the end of the line) and recognises by
// carrying the set of positions every expression can end at, which is exact
// for any grammar without left recursion. It is not a decoder: it answers only
// whether a whole input is in the language.
type gbnfRecognizer struct {
	rules map[string]*gbnfNode
}

type gbnfNode struct {
	kind     byte // 'l' literal, 'c' class, 'r' rule reference, 's' sequence, 'a' alternation, 'x' repetition
	lit      string
	ranges   [][2]rune
	negate   bool
	ref      string
	kids     []*gbnfNode
	min, max int // repetition bounds; max < 0 is unbounded
}

func parseGBNF(src string) (*gbnfRecognizer, error) {
	p := &gbnfParser{src: src}
	rules := map[string]*gbnfNode{}
	for {
		p.space(true)
		if p.pos >= len(p.src) {
			break
		}
		name := p.name()
		if name == "" {
			return nil, p.errorf("expected a rule name")
		}
		p.space(false)
		if !strings.HasPrefix(p.src[p.pos:], "::=") {
			return nil, p.errorf("expected ::= after %s", name)
		}
		p.pos += 3
		p.space(true)
		n, err := p.alternation(false)
		if err != nil {
			return nil, err
		}
		rules[name] = n
	}
	for _, n := range rules {
		if err := checkRefs(n, rules); err != nil {
			return nil, err
		}
	}
	if rules["root"] == nil {
		return nil, fmt.Errorf("gbnf: no root rule")
	}
	return &gbnfRecognizer{rules: rules}, nil
}

func checkRefs(n *gbnfNode, rules map[string]*gbnfNode) error {
	if n.kind == 'r' && rules[n.ref] == nil {
		return fmt.Errorf("gbnf: undefined rule %s", n.ref)
	}
	for _, k := range n.kids {
		if err := checkRefs(k, rules); err != nil {
			return err
		}
	}
	return nil
}

type gbnfParser struct {
	src string
	pos int
}

func (p *gbnfParser) errorf(format string, args ...any) error {
	return fmt.Errorf("gbnf at %d: %s", p.pos, fmt.Sprintf(format, args...))
}

// space skips blanks and comments, and newlines when newlines are allowed.
func (p *gbnfParser) space(newlines bool) {
	for p.pos < len(p.src) {
		switch c := p.src[p.pos]; {
		case c == ' ' || c == '\t':
			p.pos++
		case c == '#':
			for p.pos < len(p.src) && p.src[p.pos] != '\n' {
				p.pos++
			}
		case (c == '\n' || c == '\r') && newlines:
			p.pos++
		default:
			return
		}
	}
}

func (p *gbnfParser) name() string {
	start := p.pos
	for p.pos < len(p.src) {
		c := p.src[p.pos]
		if c != '-' && c != '_' && (c < '0' || c > '9') && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') {
			break
		}
		p.pos++
	}
	return p.src[start:p.pos]
}

func (p *gbnfParser) alternation(nested bool) (*gbnfNode, error) {
	alt := &gbnfNode{kind: 'a'}
	for {
		seq, err := p.sequence(nested)
		if err != nil {
			return nil, err
		}
		alt.kids = append(alt.kids, seq)
		if p.pos >= len(p.src) || p.src[p.pos] != '|' {
			return alt, nil
		}
		p.pos++
		p.space(true)
	}
}

func (p *gbnfParser) sequence(nested bool) (*gbnfNode, error) {
	seq := &gbnfNode{kind: 's'}
	for p.pos < len(p.src) {
		var n *gbnfNode
		switch c := p.src[p.pos]; {
		case c == '|' || c == ')' || c == '\n' || c == '\r':
			return seq, nil
		case c == '"':
			p.pos++
			var b strings.Builder
			for p.pos < len(p.src) && p.src[p.pos] != '"' {
				r, err := p.char()
				if err != nil {
					return nil, err
				}
				b.WriteRune(r)
			}
			if p.pos >= len(p.src) {
				return nil, p.errorf("unterminated literal")
			}
			p.pos++
			n = &gbnfNode{kind: 'l', lit: b.String()}
		case c == '[':
			p.pos++
			n = &gbnfNode{kind: 'c'}
			if p.pos < len(p.src) && p.src[p.pos] == '^' {
				n.negate = true
				p.pos++
			}
			for p.pos < len(p.src) && p.src[p.pos] != ']' {
				lo, err := p.char()
				if err != nil {
					return nil, err
				}
				hi := lo
				if p.pos+1 < len(p.src) && p.src[p.pos] == '-' && p.src[p.pos+1] != ']' {
					p.pos++
					if hi, err = p.char(); err != nil {
						return nil, err
					}
				}
				n.ranges = append(n.ranges, [2]rune{lo, hi})
			}
			if p.pos >= len(p.src) {
				return nil, p.errorf("unterminated character class")
			}
			p.pos++
		case c == '(':
			p.pos++
			p.space(true)
			var err error
			if n, err = p.alternation(true); err != nil {
				return nil, err
			}
			if p.pos >= len(p.src) || p.src[p.pos] != ')' {
				return nil, p.errorf("expected )")
			}
			p.pos++
		default:
			name := p.name()
			if name == "" {
				return nil, p.errorf("unexpected %q", c)
			}
			n = &gbnfNode{kind: 'r', ref: name}
		}
		for p.pos < len(p.src) {
			lo, hi := 0, 0
			switch p.src[p.pos] {
			case '?':
				lo, hi = 0, 1
			case '*':
				lo, hi = 0, -1
			case '+':
				lo, hi = 1, -1
			case '{':
				end := strings.IndexByte(p.src[p.pos:], '}')
				if end < 0 {
					return nil, p.errorf("unterminated repetition")
				}
				body := p.src[p.pos+1 : p.pos+end]
				var err error
				if m, n, ok := strings.Cut(body, ","); ok {
					if lo, err = strconv.Atoi(m); err == nil {
						hi = -1
						if n != "" {
							hi, err = strconv.Atoi(n)
						}
					}
				} else if lo, err = strconv.Atoi(body); err == nil {
					hi = lo
				}
				if err != nil {
					return nil, p.errorf("bad repetition {%s}", body)
				}
				p.pos += end
			default:
				lo = -1
			}
			if lo < 0 {
				break
			}
			p.pos++
			n = &gbnfNode{kind: 'x', kids: []*gbnfNode{n}, min: lo, max: hi}
		}
		seq.kids = append(seq.kids, n)
		p.space(nested)
	}
	return seq, nil
}

// char reads one character of a literal or class, decoding escapes.
func (p *gbnfParser) char() (rune, error) {
	if p.src[p.pos] != '\\' {
		r, size := utf8.DecodeRuneInString(p.src[p.pos:])
		p.pos += size
		return r, nil
	}
	if p.pos+1 >= len(p.src) {
		return 0, p.errorf("dangling escape")
	}
	c := p.src[p.pos+1]
	p.pos += 2
	switch c {
	case 'n':
		return '\n', nil
	case 't':
		return '\t', nil
	case 'r':
		return '\r', nil
	case 'x':
		if p.pos+2 > len(p.src) {
			return 0, p.errorf("short \\x escape")
		}
		v, err := strconv.ParseUint(p.src[p.pos:p.pos+2], 16, 8)
		if err != nil {
			return 0, p.errorf("bad \\x escape")
		}
		p.pos += 2
		return rune(v), nil
	default:
		return rune(c), nil
	}
}

// accepts reports whether the whole of in is in the grammar's language.
func (g *gbnfRecognizer) accepts(in string) bool {
	m := &gbnfMatch{g: g, in: in, memo: map[gbnfMemoKey][]int{}}
	return slices.Contains(m.match(&gbnfNode{kind: 'r', ref: "root"}, 0), len(in))
}

type gbnfMemoKey struct {
	rule string
	pos  int
}

type gbnfMatch struct {
	g    *gbnfRecognizer
	in   string
	memo map[gbnfMemoKey][]int
}

// match returns every position n can end at when it starts at pos.
func (m *gbnfMatch) match(n *gbnfNode, pos int) []int {
	switch n.kind {
	case 'l':
		if strings.HasPrefix(m.in[pos:], n.lit) {
			return []int{pos + len(n.lit)}
		}
	case 'c':
		if pos < len(m.in) {
			r, size := utf8.DecodeRuneInString(m.in[pos:])
			in := slices.ContainsFunc(n.ranges, func(rg [2]rune) bool { return rg[0] <= r && r <= rg[1] })
			if in != n.negate {
				return []int{pos + size}
			}
		}
	case 'r':
		key := gbnfMemoKey{n.ref, pos}
		if ends, ok := m.memo[key]; ok {
			return ends
		}
		m.memo[key] = nil // a left-recursive reference matches nothing
		ends := m.match(m.g.rules[n.ref], pos)
		m.memo[key] = ends
		return ends
	case 's':
		cur := []int{pos}
		for _, k := range n.kids {
			cur = m.each(k, cur)
			if len(cur) == 0 {
				break
			}
		}
		return cur
	case 'a':
		set := map[int]bool{}
		for _, k := range n.kids {
			for _, e := range m.match(k, pos) {
				set[e] = true
			}
		}
		return slices.Sorted(maps.Keys(set))
	case 'x':
		out := map[int]bool{}
		seen := map[int]bool{pos: true}
		if n.min == 0 {
			out[pos] = true
		}
		cur := []int{pos}
		for i := 1; n.max < 0 || i <= n.max; i++ {
			cur = m.each(n.kids[0], cur)
			if n.max < 0 {
				// Unbounded: a position reached before adds nothing new.
				cur = slices.DeleteFunc(cur, func(e int) bool { return seen[e] && i > n.min })
			}
			if len(cur) == 0 {
				break
			}
			for _, e := range cur {
				seen[e] = true
				if i >= n.min {
					out[e] = true
				}
			}
		}
		return slices.Sorted(maps.Keys(out))
	}
	return nil
}

// each matches n from every start position and returns the union of the ends.
func (m *gbnfMatch) each(n *gbnfNode, starts []int) []int {
	set := map[int]bool{}
	for _, s := range starts {
		for _, e := range m.match(n, s) {
			set[e] = true
		}
	}
	return slices.Sorted(maps.Keys(set))
}
