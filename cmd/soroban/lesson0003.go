package main

import (
	"fmt"

	"github.com/tamnd/soroban/grad"
	"github.com/tamnd/soroban/nn"
)

// lesson0003 trains three score lines with softmax and cross-entropy on
// the ice, water, steam data from lessons/0003-classification, starting
// from the zero init the lesson uses, so the table matches train.py byte
// for byte.
func lesson0003() {
	xs := []float64{-2, -1, -0.5, 0.5, 1, 2}
	labels := []int{0, 0, 1, 1, 2, 2}
	const lr = 0.1

	layer := nn.NewLayer(1, 3)

	for step := 1; step <= 300; step++ {
		losses := make([]*grad.Value, len(xs))
		for i := range xs {
			zs := layer.Forward([]*grad.Value{grad.New(xs[i])})

			// Subtract the max score before exponentiating, the softmax
			// survival rule from the lesson. Shift invariance means the
			// probabilities come out identical; treating the max as a
			// constant is safe for the same reason.
			max := zs[0].Data
			for _, z := range zs[1:] {
				if z.Data > max {
					max = z.Data
				}
			}
			sum := grad.New(0)
			exps := make([]*grad.Value, len(zs))
			for k := range zs {
				exps[k] = zs[k].Sub(grad.New(max)).Exp()
				sum = sum.Add(exps[k])
			}
			p := exps[labels[i]].Div(sum)
			losses[i] = grad.New(0).Sub(p.Log())
		}
		l := grad.Mean(losses)

		switch step {
		case 1, 2, 3, 10, 50, 300:
			fmt.Printf("step %3d  loss %.9f\n", step, l.Data)
		}

		layer.ZeroGrad()
		l.Backward()
		layer.Step(lr)
	}
}
