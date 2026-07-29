package grad

import (
	"math"
	"testing"
)

// The numbers in this file were first computed by hand in lesson 0001 and
// then confirmed by numpy and torch. If autograd disagrees with the hand
// arithmetic, autograd is wrong.

var (
	xs = []float64{1, 2, 3, 4}
	ys = []float64{3, 5, 7, 9}
)

func loss(w, b *Value) *Value {
	errs := make([]*Value, len(xs))
	for i := range xs {
		yHat := w.Mul(New(xs[i])).Add(b)
		errs[i] = yHat.Sub(New(ys[i])).Sq()
	}
	return Mean(errs)
}

func TestLesson0001Gradients(t *testing.T) {
	w, b := New(0), New(0)
	l := loss(w, b)
	l.Backward()

	if l.Data != 41 {
		t.Errorf("loss at start: got %v, want 41", l.Data)
	}
	if w.Grad != -35 {
		t.Errorf("dL/dw: got %v, want -35", w.Grad)
	}
	if b.Grad != -12 {
		t.Errorf("dL/db: got %v, want -12", b.Grad)
	}
}

func TestLesson0001ThreeSteps(t *testing.T) {
	want := []float64{41, 1.12875, 0.043271875}
	w, b := New(0), New(0)
	const lr = 0.05

	for step, wantLoss := range want {
		l := loss(w, b)
		if math.Abs(l.Data-wantLoss) > 1e-9 {
			t.Fatalf("step %d loss: got %v, want %v", step+1, l.Data, wantLoss)
		}
		w.Grad, b.Grad = 0, 0
		l.Backward()
		w.Data -= lr * w.Grad
		b.Data -= lr * b.Grad
	}
}

// A node used twice must receive gradient from both uses. This is the one
// bug every from-scratch autograd has at some point, so it gets its own test:
// f = a*b + a, so df/da = b + 1 and df/db = a.
func TestGradientAccumulation(t *testing.T) {
	a, b := New(3), New(4)
	f := a.Mul(b).Add(a)
	f.Backward()

	if f.Data != 15 {
		t.Errorf("f: got %v, want 15", f.Data)
	}
	if a.Grad != 5 {
		t.Errorf("df/da: got %v, want 5", a.Grad)
	}
	if b.Grad != 3 {
		t.Errorf("df/db: got %v, want 3", b.Grad)
	}
}

// Autograd must agree with the nudge experiment from the lesson: move the
// input a hair, recompute, divide. Slopes match to roughly the nudge size.
func TestFiniteDifference(t *testing.T) {
	const h = 1e-6
	w, b := New(0), New(0)
	l := loss(w, b)
	l.Backward()

	numeric := (loss(New(h), New(0)).Data - loss(New(0), New(0)).Data) / h
	if math.Abs(numeric-w.Grad) > 1e-3 {
		t.Errorf("dL/dw: autograd %v vs finite difference %v", w.Grad, numeric)
	}

	numeric = (loss(New(0), New(h)).Data - loss(New(0), New(0)).Data) / h
	if math.Abs(numeric-b.Grad) > 1e-3 {
		t.Errorf("dL/db: autograd %v vs finite difference %v", b.Grad, numeric)
	}
}
