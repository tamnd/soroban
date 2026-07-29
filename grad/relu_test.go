package grad

import (
	"math"
	"testing"
)

func TestReluForward(t *testing.T) {
	cases := []struct{ in, want float64 }{
		{3, 3}, {-2, 0}, {0, 0}, {0.5, 0.5}, {-0.001, 0},
	}
	for _, c := range cases {
		if got := New(c.in).Relu().Data; got != c.want {
			t.Fatalf("relu(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestReluGate(t *testing.T) {
	// Open gate: gradient passes through untouched.
	a := New(2)
	out := a.Relu().Mul(New(3))
	out.Backward()
	if a.Grad != 3 {
		t.Fatalf("open gate: got grad %v, want 3", a.Grad)
	}

	// Shut gate: gradient is blocked entirely.
	b := New(-2)
	out = b.Relu().Mul(New(3))
	out.Backward()
	if b.Grad != 0 {
		t.Fatalf("shut gate: got grad %v, want 0", b.Grad)
	}
}

// TestLesson0002Gradients builds lesson 0002's two-layer model at its fixed
// init and checks the loss and all seven gradients against the hand run.
// Every value is binary-exact, so the comparisons are exact equality.
func TestLesson0002Gradients(t *testing.T) {
	xs := []float64{-2, -1, 1, 2}
	ys := []float64{2, 1, 1, 2}

	w1, b1 := New(1), New(-0.5)
	w2, b2 := New(-1), New(-0.5)
	v1, v2, c := New(1), New(1), New(0)

	errs := make([]*Value, len(xs))
	for i := range xs {
		x := New(xs[i])
		h1 := w1.Mul(x).Add(b1).Relu()
		h2 := w2.Mul(x).Add(b2).Relu()
		yHat := v1.Mul(h1).Add(v2.Mul(h2)).Add(c)
		errs[i] = yHat.Sub(New(ys[i])).Sq()
	}
	loss := Mean(errs)
	loss.Backward()

	if loss.Data != 0.25 {
		t.Fatalf("loss = %v, want 0.25", loss.Data)
	}
	want := map[string]struct {
		v    *Value
		grad float64
	}{
		"dw1": {w1, -0.75}, "db1": {b1, -0.5},
		"dw2": {w2, 0.75}, "db2": {b2, -0.5},
		"dv1": {v1, -0.5}, "dv2": {v2, -0.5},
		"dc": {c, -1},
	}
	for name, w := range want {
		if w.v.Grad != w.grad {
			t.Fatalf("%s = %v, want %v", name, w.v.Grad, w.grad)
		}
	}
}

// TestReluFiniteDifference checks the relu gate against a nudge, away from
// the kink where the nudge test is honest.
func TestReluFiniteDifference(t *testing.T) {
	f := func(x float64) float64 {
		v := New(x)
		return v.Relu().Sq().Data
	}
	for _, x := range []float64{1.5, -1.5, 0.3} {
		v := New(x)
		out := v.Relu().Sq()
		out.Backward()
		h := 1e-6
		fd := (f(x+h) - f(x)) / h
		if math.Abs(fd-v.Grad) > 1e-4 {
			t.Fatalf("at x=%v: autograd %v vs nudge %v", x, v.Grad, fd)
		}
	}
}
