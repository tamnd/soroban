"""Builds lesson.ipynb from source.

The notebook is generated, not hand-edited, so its asserts stay in sync with
train.py and the README. The model and the ledger are inlined so it runs on
Colab with no local files. Regenerate and re-execute with:

    uv run --with nbformat lessons/0009-flops-ledger/build_notebook.py
    uv run --extra notebook python -c "import nbformat, nbclient; \
nb = nbformat.read('lessons/0009-flops-ledger/lesson.ipynb', as_version=4); \
nbclient.NotebookClient(nb, timeout=1200).execute(); \
nbformat.write(nb, 'lessons/0009-flops-ledger/lesson.ipynb')"
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

ASSETS = "https://raw.githubusercontent.com/tamnd/soroban/main/lessons/0009-flops-ledger/assets"
MATHS = "https://github.com/tamnd/soroban/blob/main/maths"

cells.append(md(
f"""# Lesson 0009: the FLOPs ledger

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tamnd/soroban/blob/main/lessons/0009-flops-ledger/lesson.ipynb)

Every lesson so far asked whether a model learns. This one asks what learning costs, in the only currency a GPU spends: floating-point operations. The whole cost of a training run is one multiplication away from known. A run over `D` tokens of a model with `N` parameters costs about `6ND` operations, and that formula comes straight out of counting the multiplies in a matrix product. This notebook derives the ledger, puts GPT-2-124M on it to the exact operation, then builds that model and times it on whatever GPU Colab gives you to measure the fraction of peak it actually reaches. The full writeup is in the [lesson README](https://github.com/tamnd/soroban/tree/main/lessons/0009-flops-ledger).

![the ledger: one matmul, to 6N per token, to 6ND for the run]({ASSETS}/ledger.png)"""))

cells.append(md(
f"""## 0. The ledger, from one matmul to a floor in days

A matrix multiply of an `m x k` and a `k x n` matrix costs `2*m*k*n` operations: `m*n` outputs, each a dot product of length `k`, about `2k` each. A forward pass runs each of the `N` parameters as one multiply-add per token, `2N`; the backward pass is two multiplies the same size, `4N`; together `6N` per token, and `6ND` over the run. GPT-2-124M has 124439808 parameters, 123653376 of them non-embedding, the `N` that goes in the rule. The full page is [flops.md]({MATHS}/flops.md)."""))

cells.append(code(
'''V, D, L, T, H = 50257, 768, 12, 1024, 12

def params(V, D, L, T):
    wte = V * D; wpe = T * D
    block = (2*D) + (D*3*D + 3*D) + (D*D + D) + (2*D) + (D*4*D + 4*D) + (4*D*D + D)
    total = wte + wpe + L*block + 2*D
    return total, total - wpe

total, non = params(V, D, L, T)
six_n = 6 * non
attn = 12 * L * H * (D // H) * T
budget = 6 * non * 3e11
floor_days = budget / 165.2e12 / 86400
print(f"params    {total} total, {non} non-embedding (the N in 6ND)")
print(f"per token 6N = {six_n}, plus attention {attn}, full {six_n + attn}")
print(f"budget    6ND over 3e11 tokens = {budget:.2e} flops")
print(f"floor     {floor_days:.1f} days at the 4090's 165.2 TFLOPs and 100% MFU")
assert (total, non) == (124439808, 123653376)
assert six_n == 741920256 and attn == 113246208
assert abs(floor_days - 15.6) < 0.1'''))

cells.append(md(
"""## 1. The exit test: GPT-2-medium by hand

Same recipe, a bigger config: vocabulary 50257, width 1024, 24 layers, context 1024, 16 heads. Predict its total and non-embedding parameter counts before you run the cell."""))

cells.append(code(
'''m_total, m_non = params(50257, 1024, 24, 1024)
print(f"GPT-2-medium: {m_total} total, {m_non} non-embedding, 6N/token {6*m_non}")
assert (m_total, m_non) == (354823168, 353774592)
assert 6 * m_non == 2122647552'''))

cells.append(md(
f"""## 2. Measuring a real GPU

Build GPT-2-124M, confirm it reports 124439808 parameters, then time forward-plus-backward and turn the stopwatch into tokens per second, achieved teraFLOPs, and MFU against the device's advertised peak. On a 4090 this is about 56 percent MFU eager and 67 percent with `torch.compile`; a free Colab T4 will show a lower number but the same shape. The cell needs a CUDA GPU and skips cleanly without one.

![measured against the 165.2 TFLOPs peak: eager, compiled, and the batch-24 cliff]({ASSETS}/roofline.png)"""))

cells.append(code(
'''import time

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    HAVE_CUDA = torch.cuda.is_available()
except ImportError:
    HAVE_CUDA = False

if not HAVE_CUDA:
    print("no CUDA GPU here, skipping the benchmark (Colab: Runtime, Change runtime type, GPU)")
else:
    PEAK = 165.2e12   # the 4090's advertised bf16 peak; change to your card's for a true MFU

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D)
            self.attn = nn.Linear(D, 3*D); self.proj = nn.Linear(D, D)
            self.fc = nn.Linear(D, 4*D); self.fc2 = nn.Linear(4*D, D)
        def forward(self, x):
            B, Tt, _ = x.shape
            h = self.ln1(x)
            q, k, v = self.attn(h).split(D, dim=2)
            q = q.view(B, Tt, H, D//H).transpose(1, 2)
            k = k.view(B, Tt, H, D//H).transpose(1, 2)
            v = v.view(B, Tt, H, D//H).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, Tt, D)
            x = x + self.proj(y)
            x = x + self.fc2(F.gelu(self.fc(self.ln2(x))))
            return x

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(V, D); self.wpe = nn.Embedding(T, D)
            self.blocks = nn.ModuleList([Block() for _ in range(L)])
            self.lnf = nn.LayerNorm(D); self.head = nn.Linear(D, V, bias=False)
            self.head.weight = self.wte.weight
        def forward(self, idx, tgt):
            pos = torch.arange(idx.shape[1], device=idx.device)
            x = self.wte(idx) + self.wpe(pos)
            for b in self.blocks: x = b(x)
            return F.cross_entropy(self.head(self.lnf(x)).view(-1, V), tgt.view(-1))

    def bench(batch):
        torch.manual_seed(0)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = GPT().cuda()
        assert sum(p.numel() for p in model.parameters()) == 124439808
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        per_token = 6*non + 12*L*H*(D//H)*T
        def step():
            idx = torch.randint(0, V, (batch, T), device="cuda")
            tgt = torch.randint(0, V, (batch, T), device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(idx, tgt)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        for _ in range(8): step()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(30): step()
        torch.cuda.synchronize(); dt = (time.time() - t0) / 30
        tok_s = batch*T/dt
        mfu = per_token*batch*T/dt / PEAK
        print(f"batch {batch}: {dt*1000:.1f} ms/iter, {tok_s:,.0f} tok/sec, {mfu*100:.1f}% MFU")
        return tok_s

    print(f"device {torch.cuda.get_device_name(0)}")
    tok_s = bench(12)
    print(f"300B tokens at this rate: {3e11/tok_s/86400:.1f} days on this GPU")'''))

cells.append(md(
"""## Exercises

1. GPT-2-medium's `6N` per token is 2122647552. If a run trains it on 300 billion tokens, what is the `6ND` budget, and how does it compare to GPT-2-124M's `2.23e20`?
2. The attention term `12*L*H*(D/H)*T` grows with context `T`. Recompute it for `T = 2048` and find its fraction of `6N`. Why does `6ND` get less accurate as context grows?
3. Raise the batch in the benchmark until the GPU runs out of memory. Note the throughput just before it fails: it falls, not rises, because the allocator is thrashing near the memory ceiling.

Worked answers are in the [lesson README](https://github.com/tamnd/soroban/tree/main/lessons/0009-flops-ledger) and asserted in `train.py`. Lesson 0010 turns from what a run costs to how to make a cheap one count: LoRA and QLoRA fine-tuning arithmetic."""))

nb.cells = cells
nbf.write(nb, str(Path(__file__).parent / "lesson.ipynb"))
print("written")
