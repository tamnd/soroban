package main

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/tamnd/soroban/attention"
)

// lesson0007 runs one causal self-attention head over three fixed tokens and
// prints the same seven-line headline as lessons/0007-attention/train.py. The
// first four lines are the head's forward pass (every weight and output forced by
// the embeddings and the causal mask); the last three are the two loss floors on
// the aba, cbc corpus that show why a bigram, keyed on one letter, cannot capture
// a dependency two letters back, and attention can.
func lesson0007() {
	E := [][]float64{
		{1, 0},
		{0, 1},
		{1, 1},
	}
	d := len(E[0])
	scale := 1.0 / math.Sqrt(float64(d))
	weights, out := attention.Head(E)

	fmt.Printf("head   1 causal self-attention head, 3 tokens, d=%d, scale 1/sqrt(2)=%.6f\n", d, scale)
	for i := range E {
		fmt.Printf("tok%d   attends %-9s w %-35s out (%.6f, %.6f)\n",
			i, attendSet(i+1), formatWeights(weights[i]), out[i][0], out[i][1])
	}

	corpus := []string{"aba", "cbc"}
	bigram := attention.BigramFloor(corpus)
	context := attention.ContextFloor(corpus)
	fmt.Println("task   corpus aba cbc: the letter after b needs the letter TWO back")
	fmt.Printf("bigram  floor log2   %.6f  (after b comes a or c, 50/50, stuck)\n", bigram)
	fmt.Printf("context floor log2/4 %.6f  (two back resolves all but the first letter)\n", context)
}

// attendSet renders the positions a token attends to as {0}, {0,1}, {0,1,2}.
func attendSet(n int) string {
	parts := make([]string, n)
	for i := range n {
		parts[i] = strconv.Itoa(i)
	}
	return "{" + strings.Join(parts, ",") + "}"
}

// formatWeights renders attention weights as space-separated six-decimal numbers,
// matching the numpy formatting in train.py.
func formatWeights(w []float64) string {
	parts := make([]string, len(w))
	for i, x := range w {
		parts[i] = strconv.FormatFloat(x, 'f', 6, 64)
	}
	return strings.Join(parts, " ")
}
