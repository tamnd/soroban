package main

import (
	"fmt"

	"github.com/tamnd/soroban/grad"
	"github.com/tamnd/soroban/nn"
)

// lesson0002 trains a two-neuron hidden layer on the V data from
// lessons/0002-hidden-layer, using the fixed init from the lesson so the
// table matches train.py byte for byte.
func lesson0002() {
	xs := []float64{-2, -1, 1, 2}
	ys := []float64{2, 1, 1, 2}
	const lr = 0.1

	hidden := nn.NewLayer(1, 2)
	hidden.Neurons[0].W[0].Data = 1
	hidden.Neurons[0].B.Data = -0.5
	hidden.Neurons[1].W[0].Data = -1
	hidden.Neurons[1].B.Data = -0.5
	out := nn.NewNeuron(2)
	out.W[0].Data = 1
	out.W[1].Data = 1

	for step := 1; step <= 300; step++ {
		errs := make([]*grad.Value, len(xs))
		for i := range xs {
			hs := hidden.Forward([]*grad.Value{grad.New(xs[i])})
			for j := range hs {
				hs[j] = hs[j].Relu()
			}
			errs[i] = out.Forward(hs).Sub(grad.New(ys[i])).Sq()
		}
		l := grad.Mean(errs)

		switch step {
		case 1, 2, 3, 10, 50, 300:
			fmt.Printf("step %3d  loss %.9f\n", step, l.Data)
		}

		hidden.ZeroGrad()
		out.ZeroGrad()
		l.Backward()
		hidden.Step(lr)
		out.Step(lr)
	}
}
