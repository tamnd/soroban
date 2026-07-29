package main

import (
	"fmt"
	"strings"

	"github.com/tamnd/soroban/bigram"
)

// lesson0005 is the counting bigram model: tally the corpus, normalize into
// next-letter probabilities, report the loss, sample new words, and show the
// zero-probability failure. The seven printed lines match
// lessons/0005-bigram/train.py byte for byte. No grad package here: this model
// is arithmetic on a table, not gradient descent.
func lesson0005() {
	corpus := []string{"cat", "cot", "cab"}
	alphabet := []rune{'.', 'a', 'b', 'c', 'o', 't'}
	m := bigram.Train(corpus, alphabet)

	fmt.Println("corpus " + strings.Join(corpus, " "))
	fmt.Println("alphabet " + spaced(alphabet))

	// Counts and probabilities, iterated in alphabet order so both languages
	// emit the same string.
	var counts, probs []string
	for _, a := range alphabet {
		for _, b := range alphabet {
			if n := m.Count(a, b); n > 0 {
				counts = append(counts, fmt.Sprintf("%c%c %d", a, b, n))
				probs = append(probs, fmt.Sprintf("%c%c %.3f", a, b, m.Prob(a, b)))
			}
		}
	}
	fmt.Println("counts " + strings.Join(counts, "  "))
	fmt.Println("probs " + strings.Join(probs, "  "))
	fmt.Printf("loss %.6f  (log3/4, every word 1/3)\n", m.NLL(corpus))

	dice := []float64{0.5, 0.2, 0.8, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.2, 0.3, 0.5}
	fmt.Println("sample " + strings.Join(m.Sample(dice, 3), " "))

	fmt.Printf("holdout dog P %.6f  loss infinite  (unseen .d)\n", m.WordProb("dog"))
}

// spaced joins runes with single spaces: ". a b c o t".
func spaced(rs []rune) string {
	parts := make([]string, len(rs))
	for i, r := range rs {
		parts[i] = string(r)
	}
	return strings.Join(parts, " ")
}
