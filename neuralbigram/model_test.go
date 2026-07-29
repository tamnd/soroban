package neuralbigram

import (
	"math"
	"testing"
)

var (
	corpus   = []string{"cat", "cot", "cab"}
	alphabet = []rune{'.', 'a', 'b', 'c', 'o', 't'}
	dice     = []float64{0.5, 0.2, 0.8, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.2, 0.3, 0.5}
)

func approx(a, b, tol float64) bool { return math.Abs(a-b) < tol }

// At zero weights every softmax row is uniform, so the loss is log 6.
func TestInitLoss(t *testing.T) {
	xs, ys := Pairs(corpus, alphabet)
	m := New(alphabet)
	if got := m.Loss(xs, ys); !approx(got, math.Log(6), 1e-12) {
		t.Fatalf("init loss = %v, want log 6 = %v", got, math.Log(6))
	}
	if got := m.Prob('c', 'a'); !approx(got, 1.0/6, 1e-12) {
		t.Fatalf("init P(a|c) = %v, want 1/6", got)
	}
}

// The one-step row-c gradient is (0.5 - observed)/12, and one step at learning
// rate 10 moves W[c,a] from 0 to 1.25. This is the by-hand core of the lesson.
func TestOneStepGradient(t *testing.T) {
	xs, ys := Pairs(corpus, alphabet)
	m := New(alphabet)
	g := m.Grad(xs, ys, 0)
	ci, ai := 3, 1
	want := []float64{
		(0.5 - 0) / 12, (0.5 - 2) / 12, (0.5 - 0) / 12,
		(0.5 - 0) / 12, (0.5 - 1) / 12, (0.5 - 0) / 12,
	}
	for j, w := range want {
		if !approx(g[ci][j], w, 1e-12) {
			t.Fatalf("grad[c][%d] = %v, want %v", j, g[ci][j], w)
		}
	}
	if !approx(g[ci][ai], -0.125, 1e-12) {
		t.Fatalf("grad[c][a] = %v, want -0.125", g[ci][ai])
	}
	if after := 0 - 10*g[ci][ai]; !approx(after, 1.25, 1e-12) {
		t.Fatalf("W[c,a] after one step = %v, want 1.25", after)
	}
}

// After 200 steps at learning rate 10, the softmax rows approach the lesson 0005
// count table and the loss its minimum log(3)/4, from above.
func TestConverges(t *testing.T) {
	xs, ys := Pairs(corpus, alphabet)
	m := New(alphabet)
	m.Train(xs, ys, 10, 200, 0)
	if got := m.Loss(xs, ys); !approx(got, 0.277674, 1e-5) {
		t.Fatalf("trained loss = %v, want 0.277674", got)
	}
	if got := m.Loss(xs, ys); got <= math.Log(3)/4 {
		t.Fatalf("trained loss %v should stay above log(3)/4 = %v", got, math.Log(3)/4)
	}
	if got := m.Prob('c', 'a'); !approx(got, 0.665334, 1e-5) {
		t.Fatalf("trained P(a|c) = %v, want 0.665334", got)
	}
	if got := m.Prob('c', 'o'); !approx(got, 0.331997, 1e-5) {
		t.Fatalf("trained P(o|c) = %v, want 0.331997", got)
	}
}

// A softmax never outputs an exact zero, so the unseen bigram c->t is small but
// positive: the neural model has no zero-frequency problem.
func TestNoZeroFrequency(t *testing.T) {
	xs, ys := Pairs(corpus, alphabet)
	m := New(alphabet)
	m.Train(xs, ys, 10, 200, 0)
	ct := m.Prob('c', 't')
	if ct <= 0 || !approx(ct, 0.000667, 1e-5) {
		t.Fatalf("trained P(t|c) = %v, want 0.000667 and positive", ct)
	}
}

// The trained table, sampled from the fixed dice, writes the training set back.
func TestSample(t *testing.T) {
	xs, ys := Pairs(corpus, alphabet)
	m := New(alphabet)
	m.Train(xs, ys, 10, 200, 0)
	got := m.Sample(dice, 3)
	want := []string{"cat", "cot", "cab"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("sample = %v, want %v", got, want)
		}
	}
}

// L2 regularization pulls the rows back toward uniform: add-one smoothing with a
// continuous dial.
func TestRegularizationSmooths(t *testing.T) {
	xs, ys := Pairs(corpus, alphabet)
	m := New(alphabet)
	m.Train(xs, ys, 10, 200, 0.10)
	if got := m.Prob('c', 'a'); !approx(got, 0.340214, 1e-5) {
		t.Fatalf("reg P(a|c) = %v, want 0.340214", got)
	}
	if got := m.Prob('c', 't'); !approx(got, 0.113314, 1e-5) {
		t.Fatalf("reg P(t|c) = %v, want 0.113314", got)
	}
}
