// Package lora counts the parameters and memory of low-rank fine-tuning. Full
// fine-tuning trains every weight; LoRA freezes the weights and trains a small
// low-rank update beside each one, and QLoRA quantizes the frozen weights to
// four bits on top of that. The whole point is a parameter and memory count that
// turns a run needing a cluster into one that fits a single 24 GB card, and it is
// the arithmetic lesson 0010 works out by hand.
package lora

// FullUpdate returns the parameter count of a full update to a d-by-k weight
// matrix, which is just its size.
func FullUpdate(d, k int) int { return d * k }

// LowRankUpdate returns the parameter count of a rank-r update B(d x r) A(r x k),
// which is r(d+k), the two thin matrices that stand in for the full d-by-k one.
func LowRankUpdate(d, k, r int) int { return r * (d + k) }

// Llama config is a decoder model's shape, enough to count its parameters.
type Llama struct {
	V int // vocabulary size
	H int // hidden width
	I int // MLP intermediate width
	L int // number of blocks
}

// Llama2With7B is the config whose parameter count is exactly 6738415616.
var Llama2With7B = Llama{V: 32000, H: 4096, I: 11008, L: 32}

// Params returns the total parameter count, block by block, with the token
// embedding and the output head untied.
func (m Llama) Params() int {
	emb := m.V * m.H
	block := 4*(m.H*m.H) + 2*(m.H*m.I) + m.I*m.H + 2*m.H // attn + mlp + two rmsnorms
	return emb + m.L*block + m.H + m.V*m.H
}

// AdapterParams returns the LoRA parameter count when a rank-r update is attached
// to the seven linear layers in every block: the four attention projections and
// the three feed-forward projections.
func (m Llama) AdapterParams(r int) int {
	perBlock := 4*LowRankUpdate(m.H, m.H, r) +
		2*LowRankUpdate(m.H, m.I, r) +
		LowRankUpdate(m.I, m.H, r)
	return m.L * perBlock
}

// FullFinetuneBytes returns the memory a full fine-tune needs, about 16 bytes per
// parameter: a half-precision weight and gradient, plus the optimizer's two
// full-precision moment estimates and a full-precision master weight.
func FullFinetuneBytes(params int) int { return params * 16 }

// QLoRABaseBytes returns the memory the frozen base needs at four bits per
// parameter, half a byte each.
func QLoRABaseBytes(params int) int { return params / 2 }
