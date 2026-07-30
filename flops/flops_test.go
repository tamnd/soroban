package flops

import (
	"math"
	"testing"
)

func approx(a, b, tol float64) bool { return math.Abs(a-b) < tol }

func TestMatmulFlops(t *testing.T) {
	// A 2-by-3 matrix times a 3-by-2 matrix: 4 outputs, each a length-3 dot
	// product, 2*2*3*2 = 24 by the 2mkn rule.
	if got := MatmulFlops(2, 3, 2); got != 24 {
		t.Fatalf("matmul flops = %d, want 24", got)
	}
}

func TestGPT2Params(t *testing.T) {
	total, non := GPT2124M.Params()
	if total != 124439808 {
		t.Fatalf("total params = %d, want 124439808", total)
	}
	if non != 123653376 {
		t.Fatalf("non-embedding params = %d, want 123653376", non)
	}
}

func TestFlopsPerToken(t *testing.T) {
	params, attn, total := GPT2124M.FlopsPerToken()
	if params != 741920256 {
		t.Fatalf("6N per token = %d, want 741920256", params)
	}
	if attn != 113246208 {
		t.Fatalf("attention per token = %d, want 113246208", attn)
	}
	if total != 855166464 {
		t.Fatalf("full per token = %d, want 855166464", total)
	}
}

func TestTrainFlops(t *testing.T) {
	_, n := GPT2124M.Params()
	// 6ND over 300 billion tokens.
	if c := TrainFlops(n, 3e11); !approx(c, 2.225760768e20, 1e12) {
		t.Fatalf("6ND = %v, want 2.225760768e20", c)
	}
}

func TestFloorDays(t *testing.T) {
	_, n := GPT2124M.Params()
	c := TrainFlops(n, 3e11)
	// At the 4090's advertised 165.2 teraFLOPs, an impossible 100 percent MFU.
	if d := FloorDays(c, 165.2e12); !approx(d, 15.6, 0.1) {
		t.Fatalf("floor days = %v, want 15.6", d)
	}
}

func TestGPT2MediumExit(t *testing.T) {
	// The exit-test config, computed here so the by-hand variant has an anchor.
	medium := Config{V: 50257, D: 1024, L: 24, T: 1024, H: 16}
	total, non := medium.Params()
	if total != 354823168 || non != 353774592 {
		t.Fatalf("medium params = (%d, %d), want (354823168, 353774592)", total, non)
	}
	if params, _, _ := medium.FlopsPerToken(); params != 2122647552 {
		t.Fatalf("medium 6N per token = %d, want 2122647552", params)
	}
}
