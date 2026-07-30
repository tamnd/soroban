// Package attention is lesson 0007's mechanism for looking further back than one
// token. A causal self-attention head lets each position read every earlier one,
// score how relevant each is by a scaled query-key dot product, and mix the
// values by those weights. This package holds the by-hand core: one head over a
// short sequence with identity query/key/value projections, so every number is
// forced by the embeddings and the causal mask alone, plus the two loss floors on
// the aba, cbc corpus that show why a bigram cannot do what attention can. The
// numbers here match lessons/0007-attention/train.py, so the Go and Python
// headlines diff empty, and the arithmetic order mirrors numpy (subtract-max
// softmax) so the printed digits agree.
package attention

import "math"

// Boundary marks the start and end of every word, as in lessons 0005 and 0006.
const Boundary = '.'

// softmax turns a row of scores into weights, subtracting the max first (the
// numerically-safe softmax from lesson 0003).
func softmax(scores []float64) []float64 {
	mx := scores[0]
	for _, s := range scores[1:] {
		if s > mx {
			mx = s
		}
	}
	out := make([]float64, len(scores))
	sum := 0.0
	for i, s := range scores {
		out[i] = math.Exp(s - mx)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// Head runs one causal self-attention head over the row-embeddings E, with the
// query, key, and value projections all set to the identity, so q = k = v = E.
// For each position i it returns the attention weights over positions j <= i and
// the output vector, the weighted sum of the earlier embeddings. The score of
// position j for position i is the dot product E[i].E[j] divided by sqrt(d),
// where d is the embedding width.
func Head(E [][]float64) (weights, out [][]float64) {
	n := len(E)
	d := len(E[0])
	scale := 1.0 / math.Sqrt(float64(d))
	weights = make([][]float64, n)
	out = make([][]float64, n)
	for i := range n {
		scores := make([]float64, i+1)
		for j := range i + 1 {
			dot := 0.0
			for c := range d {
				dot += E[i][c] * E[j][c]
			}
			scores[j] = dot * scale
		}
		w := softmax(scores)
		o := make([]float64, d)
		for j := range i + 1 {
			for c := range d {
				o[c] += w[j] * E[j][c]
			}
		}
		weights[i] = w
		out[i] = o
	}
	return weights, out
}

// wrap surrounds a word with the boundary token, as the language-model lessons do.
func wrap(w string) string {
	return string(Boundary) + w + string(Boundary)
}

// BigramFloor returns the average cross-entropy of the best possible bigram on the
// corpus: count what follows each current letter, normalize to probabilities, and
// average -log(probability of the letter that actually came next) over every
// position. On aba, cbc every current letter is followed by two different letters
// at fifty-fifty, so this is exactly log 2.
func BigramFloor(corpus []string) float64 {
	next := map[rune]map[rune]int{}
	for _, w := range corpus {
		r := []rune(wrap(w))
		for k := 0; k+1 < len(r); k++ {
			if next[r[k]] == nil {
				next[r[k]] = map[rune]int{}
			}
			next[r[k]][r[k+1]]++
		}
	}
	sum, n := 0.0, 0
	for _, w := range corpus {
		r := []rune(wrap(w))
		for k := 0; k+1 < len(r); k++ {
			row := next[r[k]]
			total := 0
			for _, c := range row {
				total += c
			}
			p := float64(row[r[k+1]]) / float64(total)
			sum += -math.Log(p)
			n++
		}
	}
	return sum / float64(n)
}

// ContextFloor returns the average cross-entropy of a model that conditions on the
// entire prefix seen so far: for each position, the probability of the next letter
// is its frequency among all continuations of that exact prefix. On aba, cbc only
// the bare prefix is ambiguous (a starts one word, c the other), and it occurs
// twice, so this is 2*log2 spread over the eight positions, i.e. log(2)/4.
func ContextFloor(corpus []string) float64 {
	cont := map[string]map[rune]int{}
	for _, w := range corpus {
		r := []rune(wrap(w))
		for k := 1; k < len(r); k++ {
			pref := string(r[:k])
			if cont[pref] == nil {
				cont[pref] = map[rune]int{}
			}
			cont[pref][r[k]]++
		}
	}
	sum, n := 0.0, 0
	for _, w := range corpus {
		r := []rune(wrap(w))
		for k := 1; k < len(r); k++ {
			row := cont[string(r[:k])]
			total := 0
			for _, c := range row {
				total += c
			}
			p := float64(row[r[k]]) / float64(total)
			sum += -math.Log(p)
			n++
		}
	}
	return sum / float64(n)
}
