package main

import (
	"fmt"

	"github.com/tamnd/soroban/flops"
)

// lesson0009 prints the same six-line headline as lessons/0009-flops-ledger/train.py.
// It is the FLOPs ledger: a matrix multiply of an m-by-k and a k-by-n matrix costs
// 2mkn operations, a forward pass is about 2N per token and a backward pass 4N, so a
// run over D tokens costs 6ND. The lines put GPT-2-124M on that ledger: its exact
// parameter count, its per-token cost including the attention term the 6N rule drops,
// the 2.23e20-operation budget for a 300-billion-token run, and the wall-clock floor
// that budget hits at an RTX 4090's advertised peak. The experiment then measures how
// far above that floor a real 4090 lands.
func lesson0009() {
	c := flops.GPT2124M
	total, non := c.Params()
	sixN, attn, per := c.FlopsPerToken()
	budget := flops.TrainFlops(non, 3e11)
	floorDays := flops.FloorDays(budget, 165.2e12)

	fmt.Println("matmul (m x k)(k x n) = 2*m*k*n flops: a multiply and an add per pair")
	fmt.Printf("params GPT-2-124M = %d total, %d non-embedding (the N in 6ND)\n", total, non)
	fmt.Printf("step   forward 2N, backward 4N, so 6N = %d flops per token\n", sixN)
	fmt.Printf("attn   extra 12*L*H*(D/H)*T = %d per token, %d per token in all\n", attn, per)
	fmt.Printf("budget 6ND over 3e11 tokens = %.2e flops to train GPT-2-124M once\n", budget)
	fmt.Printf("floor  at 4090 165.2 tflops bf16 and 100%% mfu that is %.1f days, never met\n", floorDays)
}
