// Lesson 0001 in Go: one neuron learns a line, trained with the grad and nn
// packages. The printed table matches lessons/0001-one-neuron/train.py line
// for line, which is the point: same arithmetic, different language.
package main

import (
	"fmt"

	"github.com/tamnd/soroban/grad"
	"github.com/tamnd/soroban/nn"
)

func main() {
	xs := []float64{1, 2, 3, 4}
	ys := []float64{3, 5, 7, 9}
	const lr = 0.05

	n := nn.NewNeuron(1)

	for step := 1; step <= 200; step++ {
		errs := make([]*grad.Value, len(xs))
		for i := range xs {
			yHat := n.Forward([]*grad.Value{grad.New(xs[i])})
			errs[i] = yHat.Sub(grad.New(ys[i])).Sq()
		}
		l := grad.Mean(errs)

		switch step {
		case 1, 2, 3, 10, 50, 200:
			fmt.Printf("step %3d  loss %.9f  w %.6f  b %.6f\n",
				step, l.Data, n.W[0].Data, n.B.Data)
		}

		n.ZeroGrad()
		l.Backward()
		n.Step(lr)
	}
}
