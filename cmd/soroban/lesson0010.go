package main

import (
	"fmt"

	"github.com/tamnd/soroban/lora"
)

// lesson0010 prints the same six-line headline as lessons/0010-fine-tuning/train.py.
// It is the fine-tuning ledger. A full update to a d-by-k weight is the whole matrix;
// LoRA trains a rank-r stand-in B(d x r)A(r x k) of only r(d+k) numbers, and because
// B starts at zero the adapter is a no-op at step zero so the base is untouched. The
// lines put Llama-2-7B on that ledger: its exact parameter count, the rank-16 adapter
// over the seven linears in every block and the tiny fraction it trains, and the
// memory gap between a full fine-tune that wants about 108 GB and a 4-bit QLoRA base
// that fits one 24 GB 4090. The experiment then runs that QLoRA fine-tune for real.
func lesson0010() {
	m := lora.Llama2With7B
	full := lora.FullUpdate(m.H, m.H)
	low := lora.LowRankUpdate(m.H, m.H, 16)
	total := m.Params()
	adapt := m.AdapterParams(16)
	fullGB := float64(lora.FullFinetuneBytes(total)) / 1e9
	qbaseGB := float64(lora.QLoRABaseBytes(total)) / 1e9
	qadGB := float64(lora.FullFinetuneBytes(adapt)) / 1e9

	fmt.Printf("lora   4096x4096 update: full %d, rank-16 BA = %d, %.2f%% of full\n", full, low, float64(low)/float64(full)*100)
	fmt.Println("init   B starts at 0 so BA = 0: the adapter is a no-op at step 0, base unchanged")
	fmt.Printf("model  Llama-2-7B %d params\n", total)
	fmt.Printf("adapt  lora r=16 on q,k,v,o,gate,up,down = %d trained, %.2f%%, rest frozen\n", adapt, float64(adapt)/float64(total)*100)
	fmt.Printf("memory full fine-tune ~16 bytes/param = %.1f GB, over a 24 GB 4090\n", fullGB)
	fmt.Printf("qlora  4-bit base %.2f GB + adapters %.2f GB = fits one 4090 with room\n", qbaseGB, qadGB)
}
