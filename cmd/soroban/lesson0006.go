package main

import (
	"fmt"
	"math"

	"github.com/tamnd/soroban/neuralbigram"
)

// lesson0006 trains the neural bigram on cat, cot, cab and prints the same
// seven-line headline as lessons/0006-neural-bigram/train.py. The model is a
// 6x6 weight matrix read one row at a time (an embedding), trained by full-batch
// gradient descent, and it converges to the lesson 0005 count table.
func lesson0006() {
	corpus := []string{"cat", "cot", "cab"}
	alphabet := []rune{'.', 'a', 'b', 'c', 'o', 't'}
	xs, ys := neuralbigram.Pairs(corpus, alphabet)

	m := neuralbigram.New(alphabet)
	initLoss := m.Loss(xs, ys)

	// one full-batch step, learning rate 10, to watch W[c,a] move off zero
	g := m.Grad(xs, ys, 0)
	ci := indexOf(alphabet, 'c')
	ai := indexOf(alphabet, 'a')
	wcaBefore := m.W[ci][ai]
	wcaAfter := wcaBefore - 10*g[ci][ai]

	// train from scratch: 200 steps, learning rate 10
	m.Train(xs, ys, 10, 200, 0)
	trained := m.Loss(xs, ys)
	ac := m.Prob('c', 'a')
	oc := m.Prob('c', 'o')

	dice := []float64{0.5, 0.2, 0.8, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.2, 0.3, 0.5}
	sample := m.Sample(dice, 3)

	fmt.Println("corpus cat cot cab")
	fmt.Println("model  W[current] one-hot lookup = an embedding, 6x6 = 36 weights")
	fmt.Printf("init   zeros, softmax uniform 1/6, loss %.6f  (log 6)\n", initLoss)
	fmt.Printf("grad   row c step 1: predicted minus observed, W[c,a] %.6f -> %.6f\n", wcaBefore, wcaAfter)
	fmt.Printf("train  200 steps lr 10: loss %.6f -> %.6f  (min log3/4 %.6f)\n", initLoss, trained, math.Log(3)/4)
	fmt.Printf("learned a|c %.6f  o|c %.6f  (counts %.6f %.6f)\n", ac, oc, 2.0/3, 1.0/3)
	fmt.Printf("sample %s %s %s\n", sample[0], sample[1], sample[2])
}

func indexOf(rs []rune, r rune) int {
	for i, x := range rs {
		if x == r {
			return i
		}
	}
	return -1
}
