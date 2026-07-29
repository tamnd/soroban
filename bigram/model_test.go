package bigram

import (
	"math"
	"testing"
)

// These are lesson 0005's by-hand numbers: the twelve bigrams of cat/cot/cab
// tallied and normalized, the corpus loss log(3)/4, and the sampled words the
// fixed dice produce. Counts are exact integers; probabilities are exact ratios.

var (
	corpus   = []string{"cat", "cot", "cab"}
	alphabet = []rune{'.', 'a', 'b', 'c', 'o', 't'}
)

func TestCounts(t *testing.T) {
	m := Train(corpus, alphabet)
	want := map[[2]rune]int{
		{'.', 'c'}: 3, {'c', 'a'}: 2, {'c', 'o'}: 1, {'a', 't'}: 1,
		{'a', 'b'}: 1, {'o', 't'}: 1, {'t', '.'}: 2, {'b', '.'}: 1,
	}
	total := 0
	for pair, n := range want {
		if got := m.Count(pair[0], pair[1]); got != n {
			t.Errorf("count %c%c: got %d, want %d", pair[0], pair[1], got, n)
		}
	}
	for _, a := range alphabet {
		for _, b := range alphabet {
			total += m.Count(a, b)
		}
	}
	if total != 12 {
		t.Errorf("total bigrams: got %d, want 12", total)
	}
}

func TestProbs(t *testing.T) {
	m := Train(corpus, alphabet)
	cases := []struct {
		a, b rune
		want float64
	}{
		{'.', 'c', 1}, {'c', 'a', 2.0 / 3}, {'c', 'o', 1.0 / 3},
		{'a', 't', 0.5}, {'a', 'b', 0.5}, {'o', 't', 1}, {'t', '.', 1}, {'b', '.', 1},
	}
	for _, c := range cases {
		if got := m.Prob(c.a, c.b); got != c.want {
			t.Errorf("P(%c|%c): got %v, want %v", c.b, c.a, got, c.want)
		}
	}
	// Every training word has probability exactly 1/3.
	for _, w := range corpus {
		if got := m.WordProb(w); math.Abs(got-1.0/3) > 1e-12 {
			t.Errorf("P(%s): got %v, want 1/3", w, got)
		}
	}
}

func TestLoss(t *testing.T) {
	m := Train(corpus, alphabet)
	if got := m.NLL(corpus); math.Abs(got-math.Log(3)/4) > 1e-12 {
		t.Errorf("corpus loss: got %v, want log(3)/4 = %v", got, math.Log(3)/4)
	}
}

func TestSample(t *testing.T) {
	m := Train(corpus, alphabet)
	dice := []float64{0.5, 0.2, 0.8, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.2, 0.3, 0.5}
	got := m.Sample(dice, 3)
	want := []string{"cat", "cot", "cab"}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("sample %d: got %q, want %q", i, got[i], want[i])
		}
	}
}

// TestUnseen is the deliberate failure: a word with a bigram the corpus never
// contained has probability 0, so its loss is infinite.
func TestUnseen(t *testing.T) {
	m := Train(corpus, alphabet)
	if p := m.WordProb("dog"); p != 0 {
		t.Errorf("P(dog): got %v, want 0 (unseen .d)", p)
	}
}
