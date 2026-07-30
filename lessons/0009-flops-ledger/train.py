"""Lesson 0009: the FLOPs ledger.

Default run prints the six-line headline that cmd/soroban 0009 prints byte for
byte, after asserting every number from the by-hand doc. Pass --bench to build
GPT-2-124M and time a real forward-plus-backward on a CUDA GPU, turning the
stopwatch into an MFU that reconciles with the paper.

    python train.py            # numpy asserts + headline (no torch needed)
    python train.py --bench    # add the 4090 benchmark (needs a CUDA GPU)
"""

import sys

# GPT-2-124M config: vocabulary, width, layers, context, heads.
V, D, L, T, H = 50257, 768, 12, 1024, 12
PEAK_BF16 = 165.2e12  # RTX 4090 advertised bfloat16 tensor-core throughput
TOKENS_300B = 3e11    # nanoGPT's GPT-2 reproduction trains on about this many


def params(V, D, L, T):
    """Total and non-embedding parameter counts, block by block."""
    wte = V * D
    wpe = T * D
    block = (2 * D) + (D * 3 * D + 3 * D) + (D * D + D) + (2 * D) \
        + (D * 4 * D + 4 * D) + (4 * D * D + D)
    total = wte + wpe + L * block + 2 * D
    return total, total - wpe


def headline():
    """Assert the by-hand numbers, then print the six-line ledger."""
    total, non = params(V, D, L, T)
    assert total == 124439808, total
    assert non == 123653376, non

    six_n = 6 * non
    attn = 12 * L * H * (D // H) * T
    per = six_n + attn
    assert six_n == 741920256, six_n
    assert attn == 113246208, attn
    assert per == 855166464, per

    budget = 6 * non * TOKENS_300B
    assert abs(budget - 2.225760768e20) < 1e12, budget
    floor_days = budget / PEAK_BF16 / 86400.0
    assert abs(floor_days - 15.6) < 0.1, floor_days

    # Exit test: GPT-2-medium, computed the same way.
    m_total, m_non = params(50257, 1024, 24, 1024)
    assert (m_total, m_non) == (354823168, 353774592), (m_total, m_non)
    assert 6 * m_non == 2122647552

    print("matmul (m x k)(k x n) = 2*m*k*n flops: a multiply and an add per pair")
    print(f"params GPT-2-124M = {total} total, {non} non-embedding (the N in 6ND)")
    print(f"step   forward 2N, backward 4N, so 6N = {six_n} flops per token")
    print(f"attn   extra 12*L*H*(D/H)*T = {attn} per token, {per} per token in all")
    print(f"budget 6ND over 3e11 tokens = {budget:.2e} flops to train GPT-2-124M once")
    print(f"floor  at 4090 165.2 tflops bf16 and 100% mfu that is {floor_days:.1f} days, never met")


def build_model(torch):
    """GPT-2-124M, the lesson 0007 architecture scaled up, weight-tied head."""
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(D)
            self.ln2 = nn.LayerNorm(D)
            self.attn = nn.Linear(D, 3 * D)
            self.proj = nn.Linear(D, D)
            self.fc = nn.Linear(D, 4 * D)
            self.fc2 = nn.Linear(4 * D, D)

        def forward(self, x):
            B, Tt, _ = x.shape
            h = self.ln1(x)
            q, k, v = self.attn(h).split(D, dim=2)
            q = q.view(B, Tt, H, D // H).transpose(1, 2)
            k = k.view(B, Tt, H, D // H).transpose(1, 2)
            v = v.view(B, Tt, H, D // H).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, Tt, D)
            x = x + self.proj(y)
            x = x + self.fc2(F.gelu(self.fc(self.ln2(x))))
            return x

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(V, D)
            self.wpe = nn.Embedding(T, D)
            self.blocks = nn.ModuleList([Block() for _ in range(L)])
            self.lnf = nn.LayerNorm(D)
            self.head = nn.Linear(D, V, bias=False)
            self.head.weight = self.wte.weight  # weight tying

        def forward(self, idx, targets):
            B, Tt = idx.shape
            pos = torch.arange(Tt, device=idx.device)
            x = self.wte(idx) + self.wpe(pos)
            for b in self.blocks:
                x = b(x)
            logits = self.head(self.lnf(x))
            return F.cross_entropy(logits.view(-1, V), targets.view(-1))

    return GPT()


def bench_one(torch, batch, compiled):
    """Time GPT-2-124M and return (ms/iter, tokens/sec, TFLOPs, MFU, mem GB)."""
    import time

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(torch).cuda()
    total = sum(p.numel() for p in model.parameters())
    non = total - model.wpe.weight.numel()
    assert total == 124439808, total  # the model on the scale is the paper's model
    if compiled:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
    per_token = 6 * non + 12 * L * H * (D // H) * T

    def step():
        idx = torch.randint(0, V, (batch, T), device="cuda")
        tgt = torch.randint(0, V, (batch, T), device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(idx, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(8):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    iters = 30
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    tok_s = batch * T / dt
    achieved = per_token * batch * T / dt
    mfu = achieved / PEAK_BF16
    mem = torch.cuda.max_memory_allocated() / 1e9
    return dt * 1000, tok_s, achieved / 1e12, mfu, mem


def bench():
    """Run the 4090 benchmark and reconcile the stopwatch with the ledger."""
    try:
        import torch
    except ImportError:
        print("torch not installed, skipping the benchmark")
        return
    if not torch.cuda.is_available():
        print("no CUDA GPU found, skipping the benchmark (this lesson wants a 4090)")
        return

    print(f"device {torch.cuda.get_device_name(0)}")
    print(f"{'config':<20}{'ms/iter':>9}{'tok/sec':>10}{'TFLOPs':>9}{'MFU':>8}{'mem GB':>8}")
    rows = {}
    for label, batch, compiled in [
        ("eager, batch 12", 12, False),
        ("compiled, batch 12", 12, True),
        ("eager, batch 24", 24, False),
    ]:
        ms, tok_s, tflops, mfu, mem = bench_one(torch, batch, compiled)
        rows[label] = (ms, tok_s, tflops, mfu, mem)
        print(f"{label:<20}{ms:>9.1f}{tok_s:>10,.0f}{tflops:>9.1f}{mfu*100:>7.1f}%{mem:>8.1f}")

    # The eager batch-12 run is the primary reading; reconcile it with 6ND.
    eager = rows["eager, batch 12"]
    days = TOKENS_300B / eager[1] / 86400.0
    floor = 6 * 123653376 * TOKENS_300B / PEAK_BF16 / 86400.0
    print(f"reconcile: 300B tokens at {eager[1]:,.0f} tok/sec is {days:.1f} days, "
          f"floor {floor:.1f} days at 100% mfu")

    # The deliberate failure: batch 24 is slower than batch 12, not faster.
    assert rows["eager, batch 24"][1] < eager[1], "batch 24 should collapse, not speed up"
    # Compile helps rather than hurts.
    assert rows["compiled, batch 12"][3] > eager[3], "compile should raise MFU"
    # MFU is a fraction, and eager is in a sane band for a 4090.
    assert 0.35 < eager[3] < 0.75, eager[3]


def main():
    headline()
    if "--bench" in sys.argv:
        print()
        bench()


if __name__ == "__main__":
    main()
