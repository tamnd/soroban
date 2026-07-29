// Package nn builds tiny neural networks on top of grad, growing one lesson
// at a time. As of lesson 0001 it contains exactly one thing: a neuron.
package nn

import "github.com/tamnd/soroban/grad"

// Neuron computes w1*x1 + ... + wn*xn + b. No activation function yet; that
// arrives in lesson 0002, when we first meet a problem a line cannot solve.
type Neuron struct {
	W []*grad.Value
	B *grad.Value
}

// NewNeuron returns a neuron with nin inputs and all parameters at zero.
// Zero init is deliberate here: lesson 0001 starts from total ignorance so
// every number is reproducible by hand. Lessons that need a different init
// (0002 onward, where zero init actually breaks and you get to see why)
// overwrite the fields with their fixed hand-run values.
func NewNeuron(nin int) *Neuron {
	w := make([]*grad.Value, nin)
	for i := range w {
		w[i] = grad.New(0)
	}
	return &Neuron{W: w, B: grad.New(0)}
}

// Forward returns the neuron's output for one input.
func (n *Neuron) Forward(xs []*grad.Value) *grad.Value {
	out := n.B
	for i, x := range xs {
		out = out.Add(n.W[i].Mul(x))
	}
	return out
}

// Params returns every trainable parameter.
func (n *Neuron) Params() []*grad.Value {
	return append(append([]*grad.Value{}, n.W...), n.B)
}

// ZeroGrad clears accumulated gradients. Call it before every Backward, or
// steps blend gradients from previous iterations.
func (n *Neuron) ZeroGrad() {
	for _, p := range n.Params() {
		p.Grad = 0
	}
}

// Step applies one gradient descent update: p = p - lr * dL/dp.
func (n *Neuron) Step(lr float64) {
	for _, p := range n.Params() {
		p.Data -= lr * p.Grad
	}
}
