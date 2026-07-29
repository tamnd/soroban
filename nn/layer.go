package nn

import "github.com/tamnd/soroban/grad"

// Layer is a row of neurons reading the same input. It stays linear on
// purpose: activation functions are applied by the caller, so a lesson can
// show exactly where the nonlinearity enters the computation.
type Layer struct {
	Neurons []*Neuron
}

// NewLayer returns a layer of nout neurons with nin inputs each, all
// parameters at zero. As with NewNeuron, the lessons overwrite the zeros
// with their fixed hand-run init; lesson 0002 shows why leaving a hidden
// layer at zero means it never learns at all.
func NewLayer(nin, nout int) *Layer {
	ns := make([]*Neuron, nout)
	for i := range ns {
		ns[i] = NewNeuron(nin)
	}
	return &Layer{Neurons: ns}
}

// Forward returns every neuron's output for one input.
func (l *Layer) Forward(xs []*grad.Value) []*grad.Value {
	out := make([]*grad.Value, len(l.Neurons))
	for i, n := range l.Neurons {
		out[i] = n.Forward(xs)
	}
	return out
}

// Params returns every trainable parameter in the layer.
func (l *Layer) Params() []*grad.Value {
	var ps []*grad.Value
	for _, n := range l.Neurons {
		ps = append(ps, n.Params()...)
	}
	return ps
}

// ZeroGrad clears accumulated gradients across the layer.
func (l *Layer) ZeroGrad() {
	for _, p := range l.Params() {
		p.Grad = 0
	}
}

// Step applies one gradient descent update to every parameter.
func (l *Layer) Step(lr float64) {
	for _, p := range l.Params() {
		p.Data -= lr * p.Grad
	}
}
