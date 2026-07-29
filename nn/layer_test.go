package nn

import (
	"math"
	"testing"

	"github.com/tamnd/soroban/grad"
)

// lesson0002Net builds the two-layer model from lesson 0002 at its fixed
// init: each hidden neuron owns one arm of the V, the output neuron adds
// them up.
func lesson0002Net() (*Layer, *Neuron) {
	hidden := NewLayer(1, 2)
	hidden.Neurons[0].W[0].Data = 1
	hidden.Neurons[0].B.Data = -0.5
	hidden.Neurons[1].W[0].Data = -1
	hidden.Neurons[1].B.Data = -0.5
	out := NewNeuron(2)
	out.W[0].Data = 1
	out.W[1].Data = 1
	return hidden, out
}

// TestHiddenLayerLearnsAV runs lesson 0002's training loop and pins it to
// the hand run: loss 0.25 then 0.03631953125 then 0.01158249795541518, and
// convergence with the two neurons ending as mirror images of each other.
func TestHiddenLayerLearnsAV(t *testing.T) {
	xs := []float64{-2, -1, 1, 2}
	ys := []float64{2, 1, 1, 2}
	const lr = 0.1

	hidden, out := lesson0002Net()

	var loss *grad.Value
	for step := 1; step <= 300; step++ {
		errs := make([]*grad.Value, len(xs))
		for i := range xs {
			hs := hidden.Forward([]*grad.Value{grad.New(xs[i])})
			for j := range hs {
				hs[j] = hs[j].Relu()
			}
			errs[i] = out.Forward(hs).Sub(grad.New(ys[i])).Sq()
		}
		loss = grad.Mean(errs)

		switch step {
		case 1:
			if loss.Data != 0.25 {
				t.Fatalf("step 1 loss = %v, want exactly 0.25", loss.Data)
			}
		case 2:
			if math.Abs(loss.Data-0.03631953125) > 1e-9 {
				t.Fatalf("step 2 loss = %v, want 0.03631953125", loss.Data)
			}
		case 3:
			if math.Abs(loss.Data-0.01158249795541518) > 1e-9 {
				t.Fatalf("step 3 loss = %v, want 0.011582498", loss.Data)
			}
		}

		hidden.ZeroGrad()
		out.ZeroGrad()
		loss.Backward()
		hidden.Step(lr)
		out.Step(lr)
	}

	if loss.Data > 1e-7 {
		t.Fatalf("final loss = %v, want below 1e-7", loss.Data)
	}

	n1, n2 := hidden.Neurons[0], hidden.Neurons[1]
	if math.Abs(n1.W[0].Data+n2.W[0].Data) > 1e-9 || math.Abs(n1.B.Data-n2.B.Data) > 1e-9 {
		t.Fatalf("neurons did not end as mirror images: w %v vs %v, b %v vs %v",
			n1.W[0].Data, n2.W[0].Data, n1.B.Data, n2.B.Data)
	}
}

// TestZeroInitHiddenLayerIsDead reproduces lesson 0002's first failure
// exhibit: with everything at zero, only the output bias has a gradient,
// so training drives it to mean(y) and the loss floors at 0.25.
func TestZeroInitHiddenLayerIsDead(t *testing.T) {
	xs := []float64{-2, -1, 1, 2}
	ys := []float64{2, 1, 1, 2}

	hidden := NewLayer(1, 2)
	out := NewNeuron(2)

	var loss *grad.Value
	for step := range 500 {
		errs := make([]*grad.Value, len(xs))
		for i := range xs {
			hs := hidden.Forward([]*grad.Value{grad.New(xs[i])})
			for j := range hs {
				hs[j] = hs[j].Relu()
			}
			errs[i] = out.Forward(hs).Sub(grad.New(ys[i])).Sq()
		}
		loss = grad.Mean(errs)
		hidden.ZeroGrad()
		out.ZeroGrad()
		loss.Backward()
		if step == 0 {
			for _, p := range hidden.Params() {
				if p.Grad != 0 {
					t.Fatalf("zero init: hidden gradient %v, want 0", p.Grad)
				}
			}
			if out.B.Grad != -3 {
				t.Fatalf("zero init: dc = %v, want -3", out.B.Grad)
			}
		}
		hidden.Step(0.1)
		out.Step(0.1)
	}

	if math.Abs(loss.Data-0.25) > 1e-9 || math.Abs(out.B.Data-1.5) > 1e-9 {
		t.Fatalf("zero init should floor at 0.25 with c = 1.5, got loss %v, c %v",
			loss.Data, out.B.Data)
	}
}
