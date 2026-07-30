package lora

import "testing"

// A single 4096-by-4096 attention projection: the full update is the matrix
// itself, the rank-16 update is two thin strips, under a percent of it.
func TestOneMatrix(t *testing.T) {
	full := FullUpdate(4096, 4096)
	if full != 16777216 {
		t.Fatalf("full update = %d, want 16777216", full)
	}
	low := LowRankUpdate(4096, 4096, 16)
	if low != 131072 {
		t.Fatalf("rank-16 update = %d, want 131072", low)
	}
	// The ratio is 131072 / 16777216 = 0.78 percent.
	if pct := float64(low) / float64(full) * 100; pct < 0.77 || pct > 0.79 {
		t.Fatalf("ratio = %.4f%%, want about 0.78%%", pct)
	}
}

// Llama-2-7B's parameter count from its shape, to the exact number the released
// weights carry.
func TestLlamaParams(t *testing.T) {
	if got := Llama2With7B.Params(); got != 6738415616 {
		t.Fatalf("Llama-2-7B params = %d, want 6738415616", got)
	}
}

// The rank-16 adapter over the seven linears in every block, its count and the
// fraction of the base it trains.
func TestAdapterParams(t *testing.T) {
	adapt := Llama2With7B.AdapterParams(16)
	if adapt != 39976960 {
		t.Fatalf("adapter params = %d, want 39976960", adapt)
	}
	pct := float64(adapt) / float64(Llama2With7B.Params()) * 100
	if pct < 0.58 || pct > 0.60 {
		t.Fatalf("adapter fraction = %.4f%%, want about 0.59%%", pct)
	}
	// Attention-only, the four projections in every block, is 0.249 percent of
	// the model, a quarter of the seven-linear adapter.
	attn := Llama2With7B.L * 4 * LowRankUpdate(4096, 4096, 16)
	if attn != 16777216 {
		t.Fatalf("attention-only adapter = %d, want 16777216", attn)
	}
	if apct := float64(attn) / float64(Llama2With7B.Params()) * 100; apct < 0.24 || apct > 0.26 {
		t.Fatalf("attention-only fraction = %.4f%%, want about 0.249%%", apct)
	}
}

// The memory ledger: a full fine-tune wants about 108 GB, the 4-bit frozen base
// about 3.4 GB, so one fits a 24 GB card and the other does not.
func TestMemory(t *testing.T) {
	total := Llama2With7B.Params()
	full := float64(FullFinetuneBytes(total)) / 1e9
	if full < 107.7 || full > 107.9 {
		t.Fatalf("full fine-tune = %.1f GB, want about 107.8", full)
	}
	if full <= 24 {
		t.Fatalf("full fine-tune %.1f GB should not fit a 24 GB card", full)
	}
	qbase := float64(QLoRABaseBytes(total)) / 1e9
	if qbase < 3.36 || qbase > 3.38 {
		t.Fatalf("4-bit base = %.2f GB, want about 3.37", qbase)
	}
	if qbase >= 24 {
		t.Fatalf("4-bit base %.2f GB should fit a 24 GB card", qbase)
	}
}
