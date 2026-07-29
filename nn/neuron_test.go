package nn

import (
	"math"
	"testing"

	"github.com/tamnd/soroban/grad"
)

// Train one neuron on the lesson 0001 data and check it against the numbers
// computed by hand: exact step 1, the three-step loss sequence, and where
// 200 steps land relative to the hidden truth y = 2x + 1.
func TestNeuronLearnsALine(t *testing.T) {
	xs := []float64{1, 2, 3, 4}
	ys := []float64{3, 5, 7, 9}
	const lr = 0.05

	n := NewNeuron(1)
	var lastLoss float64

	for step := 1; step <= 200; step++ {
		errs := make([]*grad.Value, len(xs))
		for i := range xs {
			yHat := n.Forward([]*grad.Value{grad.New(xs[i])})
			errs[i] = yHat.Sub(grad.New(ys[i])).Sq()
		}
		l := grad.Mean(errs)
		lastLoss = l.Data

		switch step {
		case 1:
			if l.Data != 41 {
				t.Fatalf("step 1 loss: got %v, want exactly 41", l.Data)
			}
		case 2:
			if math.Abs(l.Data-1.12875) > 1e-9 {
				t.Fatalf("step 2 loss: got %v, want 1.12875", l.Data)
			}
		case 3:
			if math.Abs(l.Data-0.043271875) > 1e-9 {
				t.Fatalf("step 3 loss: got %v, want 0.043271875", l.Data)
			}
		}

		n.ZeroGrad()
		l.Backward()
		n.Step(lr)
	}

	w, b := n.W[0].Data, n.B.Data
	if math.Abs(w-2) > 0.01 {
		t.Errorf("after 200 steps w = %v, want within 0.01 of 2", w)
	}
	if math.Abs(b-1) > 0.03 {
		t.Errorf("after 200 steps b = %v, want within 0.03 of 1", b)
	}
	if lastLoss > 1e-4 {
		t.Errorf("final loss = %v, want under 1e-4", lastLoss)
	}
}
