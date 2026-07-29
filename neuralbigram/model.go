// Package neuralbigram is lesson 0005's bigram model built a second way: instead
// of counting, it keeps a matrix of weights, one row per current symbol, scores
// each possible next symbol, runs the scores through a softmax, and trains the
// weights by gradient descent. Reading row i of the weight matrix is an embedding
// lookup. The cross-entropy gradient on the logits is predicted frequency minus
// observed frequency, so gradient descent drives the softmax rows to the observed
// frequencies, which are exactly the lesson 0005 count table. The numbers here
// match lessons/0006-neural-bigram/train.py, so the Go and Python tables diff
// empty, and the arithmetic order mirrors numpy so the trained loss agrees to the
// last printed digit.
package neuralbigram

import "math"

// Boundary marks the start and end of every word, as in lesson 0005.
const Boundary = '.'

// Model holds the weight matrix W over a fixed alphabet. W[i][j] is the score
// (logit) for symbol j following symbol i, before the softmax.
type Model struct {
	Alphabet []rune
	idx      map[rune]int
	V        int
	W        [][]float64
}

// New returns a model with all weights zero, so every softmax row starts uniform.
func New(alphabet []rune) *Model {
	v := len(alphabet)
	m := &Model{
		Alphabet: alphabet,
		idx:      make(map[rune]int, v),
		V:        v,
		W:        make([][]float64, v),
	}
	for i, r := range alphabet {
		m.idx[r] = i
		m.W[i] = make([]float64, v)
	}
	return m
}

// Pairs turns a corpus into (current, next) index examples: every bigram of every
// boundary-wrapped word, in corpus order, the same twelve pairs lesson 0005 tallied.
func Pairs(corpus []string, alphabet []rune) (xs, ys []int) {
	idx := make(map[rune]int, len(alphabet))
	for i, r := range alphabet {
		idx[r] = i
	}
	for _, w := range corpus {
		r := []rune(string(Boundary) + w + string(Boundary))
		for k := 0; k+1 < len(r); k++ {
			xs = append(xs, idx[r[k]])
			ys = append(ys, idx[r[k+1]])
		}
	}
	return xs, ys
}

// softmax turns one row of logits into probabilities, subtracting the row max
// first (the numerically-safe softmax from lesson 0003).
func softmax(logits []float64) []float64 {
	mx := logits[0]
	for _, z := range logits[1:] {
		if z > mx {
			mx = z
		}
	}
	out := make([]float64, len(logits))
	sum := 0.0
	for i, z := range logits {
		out[i] = math.Exp(z - mx)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// Prob returns P(b | a), the softmax of row a evaluated at column b.
func (m *Model) Prob(a, b rune) float64 {
	return softmax(m.W[m.idx[a]])[m.idx[b]]
}

// Row returns the full softmax distribution for the given current symbol.
func (m *Model) Row(a rune) []float64 { return softmax(m.W[m.idx[a]]) }

// Loss is the average cross-entropy over the examples: the mean of
// -log softmax(W[xs[i]])[ys[i]].
func (m *Model) Loss(xs, ys []int) float64 {
	sum := 0.0
	for i := range xs {
		p := softmax(m.W[xs[i]])
		sum += -math.Log(p[ys[i]])
	}
	return sum / float64(len(xs))
}

// Grad returns the full-batch gradient of the loss with respect to W, plus an L2
// term reg*W. For each example the logit gradient is softmax(row) minus the
// one-hot of the true next symbol, accumulated into the looked-up row and then
// averaged over all examples. Rows never looked up stay zero: the sparse
// embedding gradient.
func (m *Model) Grad(xs, ys []int, reg float64) [][]float64 {
	n := float64(len(xs))
	g := make([][]float64, m.V)
	for i := range g {
		g[i] = make([]float64, m.V)
	}
	for i := range xs {
		p := softmax(m.W[xs[i]])
		p[ys[i]] -= 1
		row := g[xs[i]]
		for j, v := range p {
			row[j] += v
		}
	}
	for i := range g {
		for j := range g[i] {
			g[i][j] = g[i][j]/n + reg*m.W[i][j]
		}
	}
	return g
}

// Step takes one full-batch gradient-descent step at the given learning rate and
// L2 strength.
func (m *Model) Step(xs, ys []int, lr, reg float64) {
	g := m.Grad(xs, ys, reg)
	for i := range m.W {
		for j := range m.W[i] {
			m.W[i][j] -= lr * g[i][j]
		}
	}
}

// Train runs steps of full-batch gradient descent, mirroring the numpy loop.
func (m *Model) Train(xs, ys []int, lr float64, steps int, reg float64) {
	for range steps {
		m.Step(xs, ys, lr, reg)
	}
}

// Sample generates nWords words by walking the softmax table exactly as the
// counting model did in lesson 0005: start at the boundary, consume one roll per
// step, pick the next symbol by inverse transform over the alphabet order, and
// stop when the boundary is drawn.
func (m *Model) Sample(dice []float64, nWords int) []string {
	words := make([]string, 0, nWords)
	d := 0
	for range nWords {
		cur := Boundary
		out := []rune{}
		for {
			u := dice[d]
			d++
			row := softmax(m.W[m.idx[cur]])
			cum := 0.0
			pick := Boundary
			for j, s := range m.Alphabet {
				cum += row[j]
				if cum >= u {
					pick = s
					break
				}
			}
			if pick == Boundary {
				break
			}
			out = append(out, pick)
			cur = pick
		}
		words = append(words, string(out))
	}
	return words
}
