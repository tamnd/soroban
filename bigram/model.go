// Package bigram is a counting language model: it tallies which symbol follows
// which in a corpus, normalizes the tallies into next-symbol probabilities, and
// samples new sequences from them. There is no gradient here, only counting and
// dividing, which is lesson 0005's whole point. The numbers it produces are the
// same ones lessons/0005-bigram/train.py asserts, so the Go and Python tables
// diff empty.
package bigram

import "math"

// Boundary is the symbol that marks the start and end of every word.
const Boundary = '.'

// Model holds the count matrix N and its row sums for a fixed alphabet. N[i][j]
// is how many times symbol j followed symbol i in the training corpus.
type Model struct {
	Alphabet []rune
	idx      map[rune]int
	N        [][]int
	rows     []int
}

// Train tallies every bigram in the corpus over the given alphabet order. Each
// word is wrapped in the boundary symbol before counting.
func Train(corpus []string, alphabet []rune) *Model {
	v := len(alphabet)
	m := &Model{
		Alphabet: alphabet,
		idx:      make(map[rune]int, v),
		N:        make([][]int, v),
		rows:     make([]int, v),
	}
	for i, r := range alphabet {
		m.idx[r] = i
		m.N[i] = make([]int, v)
	}
	for _, w := range corpus {
		s := string(Boundary) + w + string(Boundary)
		r := []rune(s)
		for k := 0; k+1 < len(r); k++ {
			a, b := m.idx[r[k]], m.idx[r[k+1]]
			m.N[a][b]++
			m.rows[a]++
		}
	}
	return m
}

// Count returns how many times symbol b followed symbol a.
func (m *Model) Count(a, b rune) int { return m.N[m.idx[a]][m.idx[b]] }

// Prob returns P(b | a), the fraction of times b followed a. A symbol that never
// occurred as a current symbol, or a pair never seen, gives 0.
func (m *Model) Prob(a, b rune) float64 {
	i := m.idx[a]
	if m.rows[i] == 0 {
		return 0
	}
	return float64(m.N[i][m.idx[b]]) / float64(m.rows[i])
}

// WordProb is the model's probability for one whole word: the product of its
// bigram probabilities, boundaries included.
func (m *Model) WordProb(word string) float64 {
	r := []rune(string(Boundary) + word + string(Boundary))
	p := 1.0
	for k := 0; k+1 < len(r); k++ {
		p *= m.Prob(r[k], r[k+1])
	}
	return p
}

// NLL is the average over every bigram in the corpus of -log P(next | current),
// the same cross-entropy loss lesson 0003 used, read off the count table.
func (m *Model) NLL(corpus []string) float64 {
	sum, n := 0.0, 0
	for _, w := range corpus {
		r := []rune(string(Boundary) + w + string(Boundary))
		for k := 0; k+1 < len(r); k++ {
			sum += -math.Log(m.Prob(r[k], r[k+1]))
			n++
		}
	}
	return sum / float64(n)
}

// Sample generates nWords words by walking the table: start at the boundary, and
// at each step consume one roll from dice and pick the next symbol by inverse
// transform over the alphabet order, stopping when the boundary is drawn. The
// fixed dice make the output identical to the Python sampler.
func (m *Model) Sample(dice []float64, nWords int) []string {
	words := make([]string, 0, nWords)
	d := 0
	for range nWords {
		cur := Boundary
		out := []rune{}
		for {
			u := dice[d]
			d++
			cum := 0.0
			pick := Boundary
			for _, s := range m.Alphabet {
				cum += m.Prob(cur, s)
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
