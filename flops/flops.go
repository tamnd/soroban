// Package flops counts the floating-point operations in a transformer training
// run. It is the arithmetic behind the 6ND rule: a run over D tokens of a model
// with N parameters costs about 6ND operations, which is enough to budget a run
// before it starts. Everything here is integer counting and one division, the
// same numbers lesson 0009 works out by hand.
package flops

// Config is a GPT-2-style transformer's shape.
type Config struct {
	V int // vocabulary size
	D int // embedding width
	L int // number of transformer blocks
	T int // context length
	H int // number of attention heads
}

// GPT2124M is the config of the 124M-parameter GPT-2 that lesson 0009 measures.
var GPT2124M = Config{V: 50257, D: 768, L: 12, T: 1024, H: 12}

// MatmulFlops returns the cost of multiplying an m-by-k matrix with a k-by-n
// matrix. Each of the m*n outputs is a dot product of length k, about 2k
// operations, so the whole multiply is 2*m*k*n.
func MatmulFlops(m, k, n int) int { return 2 * m * k * n }

// blockParams counts the trainable numbers in one transformer block: two
// layernorms, the query-key-value and output projections, and the two
// feed-forward projections, each matmul weight carrying a bias.
func blockParams(D int) int {
	ln1 := 2 * D
	cAttn := D*(3*D) + 3*D
	cProj := D*D + D
	ln2 := 2 * D
	mlpUp := D*(4*D) + 4*D
	mlpDown := (4*D)*D + D
	return ln1 + cAttn + cProj + ln2 + mlpUp + mlpDown
}

// Params returns the total parameter count and the non-embedding count. The
// non-embedding count drops the position table, which is looked up rather than
// multiplied through, and it is the N that goes in 6ND. The output head shares
// weights with the token embedding, so it adds nothing.
func (c Config) Params() (total, nonEmbedding int) {
	wte := c.V * c.D
	wpe := c.T * c.D
	total = wte + wpe + c.L*blockParams(c.D) + 2*c.D
	nonEmbedding = total - wpe
	return
}

// FlopsPerToken returns the forward-plus-backward cost of one token: 6N for the
// parameter matmuls, plus the attention-score term that grows with context
// length rather than with parameters. The total is the honest per-token cost.
func (c Config) FlopsPerToken() (params, attn, total int) {
	_, n := c.Params()
	params = 6 * n
	attn = 12 * c.L * c.H * (c.D / c.H) * c.T
	total = params + attn
	return
}

// TrainFlops returns 6ND, the operation count to train a model with n
// non-embedding parameters over the given number of tokens.
func TrainFlops(n int, tokens float64) float64 {
	return 6 * float64(n) * tokens
}

// FloorDays returns the wall-clock days a FLOP budget would take at a device's
// advertised peak rate, an unreachable floor because no real run sustains its
// peak. peakFlops is operations per second.
func FloorDays(totalFlops, peakFlops float64) float64 {
	return totalFlops / peakFlops / 86400.0
}
