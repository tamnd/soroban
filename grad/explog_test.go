package grad

import (
	"math"
	"testing"
)

func TestExpLogForward(t *testing.T) {
	if got := New(0).Exp().Data; got != 1 {
		t.Fatalf("exp(0) = %v, want 1", got)
	}
	if got := New(2).Exp().Data; math.Abs(got-7.38905609893065) > 1e-14 {
		t.Fatalf("exp(2) = %v, want 7.38905609893065", got)
	}
	if got := New(1).Log().Data; got != 0 {
		t.Fatalf("log(1) = %v, want 0", got)
	}
	if got := New(3).Log().Data; math.Abs(got-1.0986122886681098) > 1e-14 {
		t.Fatalf("log(3) = %v, want ln 3", got)
	}
	if got := New(1).Div(New(4)).Data; got != 0.25 {
		t.Fatalf("1/4 = %v, want 0.25", got)
	}
}

func TestExpLogInverse(t *testing.T) {
	// log undoes exp: d(log(exp(x)))/dx must be 1, since the slopes e^x
	// and 1/e^x meet in the chain rule and cancel.
	for _, x := range []float64{-1.5, 0.3, 2.0} {
		v := New(x)
		out := v.Exp().Log()
		out.Backward()
		if math.Abs(out.Data-x) > 1e-14 {
			t.Fatalf("log(exp(%v)) = %v", x, out.Data)
		}
		if math.Abs(v.Grad-1) > 1e-14 {
			t.Fatalf("slope of log(exp(x)) at %v = %v, want 1", x, v.Grad)
		}
	}
}

// TestExpLogDivFiniteDifference checks each new op against a nudge, the
// same three-way discipline the lessons run on every gradient.
func TestExpLogDivFiniteDifference(t *testing.T) {
	cases := []struct {
		name  string
		build func(*Value) *Value
		f     func(float64) float64
		at    []float64
	}{
		{"exp", func(v *Value) *Value { return v.Exp() }, math.Exp, []float64{-1, 0.5, 2}},
		{"log", func(v *Value) *Value { return v.Log() }, math.Log, []float64{0.3, 1, 4}},
		{
			"div",
			func(v *Value) *Value { return New(2).Div(v) },
			func(x float64) float64 { return 2 / x },
			[]float64{0.5, 1.5, 3},
		},
	}
	for _, c := range cases {
		for _, x := range c.at {
			v := New(x)
			out := c.build(v)
			out.Backward()
			h := 1e-6
			fd := (c.f(x+h) - c.f(x-h)) / (2 * h)
			if math.Abs(fd-v.Grad) > 1e-4 {
				t.Fatalf("%s at x=%v: autograd %v vs nudge %v", c.name, x, v.Grad, fd)
			}
		}
	}
}

// TestLesson0003Gradients builds lesson 0003's softmax cross-entropy at
// zero init and checks the loss and all six gradients against the hand
// run. Thirds and ln 3 are not binary-exact, so unlike 0001 and 0002 the
// comparisons carry a tolerance even at step 1.
func TestLesson0003Gradients(t *testing.T) {
	xs := []float64{-2, -1, -0.5, 0.5, 1, 2}
	labels := []int{0, 0, 1, 1, 2, 2}

	w := []*Value{New(0), New(0), New(0)}
	b := []*Value{New(0), New(0), New(0)}

	losses := make([]*Value, len(xs))
	for i := range xs {
		x := New(xs[i])
		exps := make([]*Value, 3)
		sum := New(0)
		for k := range exps {
			exps[k] = w[k].Mul(x).Add(b[k]).Exp()
			sum = sum.Add(exps[k])
		}
		p := exps[labels[i]].Div(sum)
		losses[i] = New(0).Sub(p.Log())
	}
	loss := Mean(losses)
	loss.Backward()

	if math.Abs(loss.Data-math.Log(3)) > 1e-12 {
		t.Fatalf("loss = %v, want ln 3", loss.Data)
	}
	wantW := []float64{0.5, 0, -0.5}
	for k := range w {
		if math.Abs(w[k].Grad-wantW[k]) > 1e-12 {
			t.Fatalf("dw%d = %v, want %v", k, w[k].Grad, wantW[k])
		}
		if math.Abs(b[k].Grad) > 1e-12 {
			t.Fatalf("db%d = %v, want 0", k, b[k].Grad)
		}
	}
}
