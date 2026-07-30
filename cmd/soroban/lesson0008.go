package main

import (
	"fmt"

	"github.com/tamnd/soroban/overfit"
)

// lesson0008 prints the same six-line headline as lessons/0008-training-loop/train.py.
// It is the by-hand core of the training loop as an instrument: fit three noisy
// points with a line and with a parabola, and read off the training and held-out
// errors of each. The parabola drives its training error to zero by memorizing the
// noise and then predicts the held-out points terribly; the line keeps a small
// training error and generalizes. The last two lines count lesson 0007's
// transformer and its parameters-per-character ratio, the same widening gap at the
// scale of a real model, which the experiment then measures directly.
func lesson0008() {
	// True line y = 0.5x + 0.5, sampled at x = 0, 1, 2 with the middle point nudged
	// off the line, and two clean held-out points on the line at x = 3, 4.
	trainX := []float64{0, 1, 2}
	trainY := []float64{0.5, 1.3, 1.5}
	valX := []float64{3, 4}
	valY := []float64{2.0, 2.5}

	line := overfit.FitPoly(trainX, trainY, 1)
	parab := overfit.FitPoly(trainX, trainY, 2)
	params := overfit.ParamCount(33, 64, 64)

	fmt.Println("fit    3 train points off a line, 2 held out; true line y=0.5x+0.5")
	fmt.Printf("line   2 params: train mse %.6f, val mse %.6f  (generalizes)\n",
		overfit.MSE(line, trainX, trainY), overfit.MSE(line, valX, valY))
	fmt.Printf("parab  3 params: train mse %.6f, val mse %.6f  (memorized the noise)\n",
		overfit.MSE(parab, trainX, trainY), overfit.MSE(parab, valX, valY))
	fmt.Println("read   lowest train mse picks parab; the val set exposes it as the worse model")
	fmt.Printf("count  transformer %d params on 9311 chars = %.3f params per char\n",
		params, float64(params)/9311.0)
	fmt.Println("regime params exceed data points, so memorizing is possible: watch held-out loss")
}
