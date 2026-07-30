package overfit

import (
	"math"
	"testing"
)

// The by-hand dataset: a true line y = 0.5x + 0.5 sampled at x = 0, 1, 2 with the
// middle point nudged off the line (noise), and two clean held-out points on the
// line at x = 3, 4. The line fit ignores the wiggle; the parabola bends to hit it.
var (
	trainX = []float64{0, 1, 2}
	trainY = []float64{0.5, 1.3, 1.5}
	valX   = []float64{3, 4}
	valY   = []float64{2.0, 2.5}
)

func approx(a, b, tol float64) bool { return math.Abs(a-b) < tol }

func TestLineCoefficients(t *testing.T) {
	// Least-squares line through the three training points is y = 0.5x + 0.6.
	c := FitPoly(trainX, trainY, 1)
	if !approx(c[0], 0.6, 1e-12) || !approx(c[1], 0.5, 1e-12) {
		t.Fatalf("line coef = %v, want [0.6 0.5]", c)
	}
}

func TestParabolaCoefficients(t *testing.T) {
	// Interpolating parabola through the three points is y = -0.3x^2 + 1.1x + 0.5.
	c := FitPoly(trainX, trainY, 2)
	if !approx(c[0], 0.5, 1e-9) || !approx(c[1], 1.1, 1e-9) || !approx(c[2], -0.3, 1e-9) {
		t.Fatalf("parabola coef = %v, want [0.5 1.1 -0.3]", c)
	}
}

func TestLineMSE(t *testing.T) {
	c := FitPoly(trainX, trainY, 1)
	if tr := MSE(c, trainX, trainY); !approx(tr, 0.02, 1e-12) {
		t.Fatalf("line train mse = %v, want 0.02", tr)
	}
	if v := MSE(c, valX, valY); !approx(v, 0.01, 1e-12) {
		t.Fatalf("line val mse = %v, want 0.01", v)
	}
}

func TestParabolaMSE(t *testing.T) {
	c := FitPoly(trainX, trainY, 2)
	// The parabola interpolates the training points exactly: training error zero.
	if tr := MSE(c, trainX, trainY); !approx(tr, 0.0, 1e-9) {
		t.Fatalf("parabola train mse = %v, want 0", tr)
	}
	// But it pays 3.285 on the held-out points it bent away from.
	if v := MSE(c, valX, valY); !approx(v, 3.285, 1e-9) {
		t.Fatalf("parabola val mse = %v, want 3.285", v)
	}
}

func TestOverfitSignature(t *testing.T) {
	// The instrument reading: the parabola wins on training error but loses badly
	// on held-out error. Training error alone would pick the worse model.
	line := FitPoly(trainX, trainY, 1)
	parab := FitPoly(trainX, trainY, 2)
	if MSE(parab, trainX, trainY) >= MSE(line, trainX, trainY) {
		t.Fatal("parabola should have the lower training error")
	}
	if MSE(parab, valX, valY) <= MSE(line, valX, valY) {
		t.Fatal("parabola should have the higher held-out error")
	}
}

func TestParamCount(t *testing.T) {
	// Lesson 0007's transformer: vocabulary 33, width 64, context 64.
	if p := ParamCount(33, 64, 64); p != 58273 {
		t.Fatalf("param count = %d, want 58273", p)
	}
}

func TestParamsPerChar(t *testing.T) {
	// 58273 parameters over 9311 training characters: more than six per character,
	// deep in the regime where memorizing the training text is possible.
	ratio := float64(ParamCount(33, 64, 64)) / 9311.0
	if !approx(ratio, 6.258511, 1e-6) {
		t.Fatalf("params per char = %v, want 6.258511", ratio)
	}
}
