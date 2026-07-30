package attention

import (
	"math"
	"testing"
)

// approx reports whether a and b are within tol. Named approx, not close, so it
// does not shadow the builtin close.
func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// the three-token embeddings from the by-hand core: dot products come out whole.
func headEmbeddings() [][]float64 {
	return [][]float64{
		{1, 0},
		{0, 1},
		{1, 1},
	}
}

func TestHeadWeights(t *testing.T) {
	w, _ := Head(headEmbeddings())
	// token 0 attends only to itself
	if len(w[0]) != 1 || !approx(w[0][0], 1.0, 1e-9) {
		t.Fatalf("token 0 weights = %v, want [1]", w[0])
	}
	// token 1 attends to positions 0 and 1
	want1 := []float64{0.330238, 0.669762}
	for j, x := range want1 {
		if !approx(w[1][j], x, 1e-6) {
			t.Fatalf("token 1 weight %d = %.6f, want %.6f", j, w[1][j], x)
		}
	}
	// token 2 attends to all three
	want2 := []float64{0.248255, 0.248255, 0.503490}
	for j, x := range want2 {
		if !approx(w[2][j], x, 1e-6) {
			t.Fatalf("token 2 weight %d = %.6f, want %.6f", j, w[2][j], x)
		}
	}
}

func TestHeadOutputs(t *testing.T) {
	_, out := Head(headEmbeddings())
	want := [][]float64{
		{1.000000, 0.000000},
		{0.330238, 0.669762},
		{0.751745, 0.751745},
	}
	for i := range want {
		for c := range want[i] {
			if !approx(out[i][c], want[i][c], 1e-6) {
				t.Fatalf("output[%d][%d] = %.6f, want %.6f", i, c, out[i][c], want[i][c])
			}
		}
	}
}

func TestHeadWeightsSumToOne(t *testing.T) {
	w, _ := Head(headEmbeddings())
	for i := range w {
		sum := 0.0
		for _, x := range w[i] {
			sum += x
		}
		if !approx(sum, 1.0, 1e-12) {
			t.Fatalf("token %d weights sum to %.12f, want 1", i, sum)
		}
	}
}

func TestBigramFloor(t *testing.T) {
	got := BigramFloor([]string{"aba", "cbc"})
	if !approx(got, math.Log(2), 1e-12) {
		t.Fatalf("bigram floor = %.12f, want log2 = %.12f", got, math.Log(2))
	}
}

func TestContextFloor(t *testing.T) {
	got := ContextFloor([]string{"aba", "cbc"})
	if !approx(got, math.Log(2)/4, 1e-12) {
		t.Fatalf("context floor = %.12f, want log2/4 = %.12f", got, math.Log(2)/4)
	}
}

// The floors must order the way the lesson claims: a bigram is strictly worse than
// a model that sees the whole prefix.
func TestFloorsOrdered(t *testing.T) {
	bg := BigramFloor([]string{"aba", "cbc"})
	ctx := ContextFloor([]string{"aba", "cbc"})
	if !(ctx < bg) {
		t.Fatalf("context floor %.6f should be below bigram floor %.6f", ctx, bg)
	}
}
