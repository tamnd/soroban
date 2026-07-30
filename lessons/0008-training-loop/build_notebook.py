"""Builds lesson.ipynb from source.

The notebook is generated, not hand-edited, so its markdown and asserts stay in
sync with train.py and the README. The TinyStories text and the transformer are
inlined into the notebook so it runs on Colab with no local files. Regenerate and
re-execute with:

    uv run --with nbformat lessons/0008-training-loop/build_notebook.py
    uv run --extra notebook python -c "import nbformat, nbclient; \
nb = nbformat.read('lessons/0008-training-loop/lesson.ipynb', as_version=4); \
nbclient.NotebookClient(nb, timeout=600).execute(); \
nbformat.write(nb, 'lessons/0008-training-loop/lesson.ipynb')"
"""

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).parent))
import train  # noqa: E402  imported for its embedded TinyStories corpus

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

ASSETS = "https://raw.githubusercontent.com/tamnd/soroban/main/lessons/0008-training-loop/assets"
MATHS = "https://github.com/tamnd/soroban/blob/main/maths"

cells.append(md(
f"""# Lesson 0008: the training loop as instrument

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tamnd/soroban/blob/main/lessons/0008-training-loop/lesson.ipynb)

Every lesson so far judged a model by its training loss and cheered when it fell. This one shows why that number, alone, lies. A model with more parameters than data points can drive its training loss to zero by memorizing, and a memorized answer key says nothing about unseen questions. The fix is a second number: the loss on held-out data. This notebook first fits three noisy points with a line and a parabola to see the gap by hand, then splits the TinyStories text and trains lesson 0007's transformer at three widths to watch the same gap open at scale. The full writeup is in the [lesson README](https://github.com/tamnd/soroban/tree/main/lessons/0008-training-loop).

![same points, two fits: the parabola memorizes and then misses]({ASSETS}/overfit.png)"""))

cells.append(md(
f"""## 0. Two fits to the same points

The true relationship is the line `y = 0.5x + 0.5`, sampled at `x = 0, 1, 2` with the middle point nudged off the line (noise), and two clean held-out points at `x = 3, 4`. A line has two parameters and keeps a small training error. A parabola has three parameters, one per training point, so it interpolates them exactly, training error zero, then bends away from the held-out points. Training error alone would pick the parabola, the worse model. The full page is [overfitting.md]({MATHS}/overfitting.md)."""))

cells.append(code(
"""import numpy as np

TRAIN_X = np.array([0.0, 1.0, 2.0]); TRAIN_Y = np.array([0.5, 1.3, 1.5])
VAL_X = np.array([3.0, 4.0]);        VAL_Y = np.array([2.0, 2.5])

def mse(coef, x, y): return float(np.mean((np.polyval(coef, x) - y) ** 2))

line = np.polyfit(TRAIN_X, TRAIN_Y, 1)      # y = 0.5x + 0.6
parab = np.polyfit(TRAIN_X, TRAIN_Y, 2)     # y = -0.3x^2 + 1.1x + 0.5
print(f"line   train mse {mse(line, TRAIN_X, TRAIN_Y):.3f}   held-out mse {mse(line, VAL_X, VAL_Y):.3f}")
print(f"parab  train mse {mse(parab, TRAIN_X, TRAIN_Y):.3f}   held-out mse {mse(parab, VAL_X, VAL_Y):.3f}")
assert abs(mse(parab, TRAIN_X, TRAIN_Y)) < 1e-9        # memorized: train error zero
assert abs(mse(parab, VAL_X, VAL_Y) - 3.285) < 1e-9    # and 328x worse held out"""))

cells.append(md(
"""## 1. The threshold is parameters against data points

A polynomial with as many parameters as data points can pass through all of them, whatever the data: three parameters memorize three points. That ratio scales to the transformer. Lesson 0007's model has 58273 parameters and trained on 9311 characters, about six parameters per character, well past the point where memorizing is easy. That, not understanding, is why its training loss fell so far."""))

cells.append(code(
"""def param_count(V, D, T):
    return (V*D + T*D + 3*(D*D) + (D*D+D)
            + (D*4*D + 4*D) + (4*D*D + D) + 3*(2*D) + (D*V + V))

P = param_count(33, 64, 64)
print(f"{P} params on 9311 chars = {P/9311:.3f} params per char")
assert P == 58273"""))

cells.append(md(
f"""## 2. The same instrument on the transformer

Split the text into 90 percent training and 10 percent held out, and train the same architecture at widths `D = 16, 32, 64` for three thousand steps each, measuring both losses every 150 steps. Wider models push training loss down and held-out loss up: the gap is overfitting. The held-out loss traces a U, bottoming early and then climbing; the bottom is where you should stop. (This cell needs torch, which Colab has preinstalled.)

![train loss keeps falling; val loss turns back up]({ASSETS}/gap.png)"""))

cells.append(code(
f'''CORPUS = """{train.CORPUS}"""

try:
    import math, torch, torch.nn.functional as F

    chars = sorted(set(CORPUS)); V = len(chars)
    stoi = {{c: i for i, c in enumerate(chars)}}
    full = torch.tensor([stoi[c] for c in CORPUS], dtype=torch.long)
    cut = int(0.9 * len(full))
    train_data, val_data = full[:cut], full[cut:]
    B, T = 32, 64
    print(f"split {{len(train_data)}} train chars, {{len(val_data)}} val chars, vocab {{V}}")

    def build(D):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tok = torch.nn.Embedding(V, D); self.pos = torch.nn.Embedding(T, D)
                self.q = torch.nn.Linear(D, D, bias=False)
                self.k = torch.nn.Linear(D, D, bias=False)
                self.v = torch.nn.Linear(D, D, bias=False)
                self.proj = torch.nn.Linear(D, D)
                self.ln1 = torch.nn.LayerNorm(D); self.ln2 = torch.nn.LayerNorm(D)
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(D, 4*D), torch.nn.GELU(), torch.nn.Linear(4*D, D))
                self.lnf = torch.nn.LayerNorm(D); self.head = torch.nn.Linear(D, V)
                self.register_buffer("mask", torch.tril(torch.ones(T, T)))
            def forward(self, idx):
                Tt = idx.shape[1]
                x = self.tok(idx) + self.pos(torch.arange(Tt)); h = self.ln1(x)
                att = (self.q(h) @ self.k(h).transpose(-2, -1)) / math.sqrt(D)
                att = att.masked_fill(self.mask[:Tt, :Tt] == 0, float("-inf"))
                att = F.softmax(att, dim=-1)
                x = x + self.proj(att @ self.v(h)); x = x + self.mlp(self.ln2(x))
                return self.head(self.lnf(x))
        return Block()

    def get_batch(data, g):
        ix = torch.randint(0, len(data)-T-1, (B,), generator=g)
        return (torch.stack([data[i:i+T] for i in ix]),
                torch.stack([data[i+1:i+T+1] for i in ix]))

    def run(D):
        torch.manual_seed(1337); model = build(D)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(1337)
        np_ = sum(p.numel() for p in model.parameters())
        @torch.no_grad()
        def est(data):
            ge = torch.Generator().manual_seed(0)
            losses = []
            for _ in range(50):
                x, y = get_batch(data, ge)
                losses.append(F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item())
            return sum(losses) / len(losses)
        best, bs, ft, fv = 1e9, 0, 0.0, 0.0
        for step in range(3001):
            if step % 150 == 0:
                tl, vl = est(train_data), est(val_data)
                if vl < best: best, bs = vl, step
                if step == 3000: ft, fv = tl, vl
            if step < 3000:
                x, y = get_batch(train_data, g)
                loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
        print(f"  D={{D:2d}}  {{np_:5d}} params  final train {{ft:.3f}}  final val {{fv:.3f}}  best val {{best:.3f}} @ step {{bs}}")
        return ft, fv, best

    rows = [run(D) for D in (16, 32, 64)]
    assert rows[0][0] > rows[2][0]      # more params -> lower training loss
    assert rows[0][1] < rows[2][1]      # more params -> higher held-out loss
    assert rows[2][2] < 2.1651 < rows[2][1]   # early-stopped beats bigram; overtrained loses
    print("early-stopped D=64 beats the bigram 2.1651; trained to the end it loses to it")
except ImportError:
    print("torch not installed, skipping the training cell (Colab has it preinstalled)")'''))

cells.append(md(
"""## Exercises

1. Which width has the lowest training loss at step 3000, and which has the lowest held-out loss? They are different models. Predict, then read the printout.
2. Recompute the parabola's held-out error by hand: evaluate `y = -0.3x^2 + 1.1x + 0.5` at `x = 3, 4`, subtract the true `2.0, 2.5`, square, average. You should get 3.285.
3. The held-out minimum arrives at step 1500 for `D = 16` and step 600 for `D = 64`. Why does more capacity make it arrive sooner?

Worked answers are in the [lesson README](https://github.com/tamnd/soroban/tree/main/lessons/0008-training-loop) and asserted in `train.py`. Lesson 0009 turns from what a model learns to what it costs to make it learn: the FLOPs ledger."""))

nb.cells = cells
nbf.write(nb, str(Path(__file__).parent / "lesson.ipynb"))
print("written")
