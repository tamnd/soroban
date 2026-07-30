"""Builds lesson.ipynb from source.

The notebook is generated, not hand-edited, so its markdown and asserts stay in
sync with train.py and the README. The TinyStories corpus and the transformer are
inlined into the notebook so it runs on Colab with no local files. Regenerate and
re-execute with:

    uv run --with nbformat lessons/0007-attention/build_notebook.py
    uv run --extra notebook python -c "import nbformat, nbclient; \
nb = nbformat.read('lessons/0007-attention/lesson.ipynb', as_version=4); \
nbclient.NotebookClient(nb, timeout=300).execute(); \
nbformat.write(nb, 'lessons/0007-attention/lesson.ipynb')"
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

ASSETS = "https://raw.githubusercontent.com/tamnd/soroban/main/lessons/0007-attention/assets"
MATHS = "https://github.com/tamnd/soroban/blob/main/maths"

cells.append(md(
f"""# Lesson 0007: attention, looking further back than one token

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tamnd/soroban/blob/main/lessons/0007-attention/lesson.ipynb)

Every model so far has had a one-token memory: the bigram and the neural bigram both predict the next letter from the current letter alone. Attention removes that wall. It lets each position read every earlier position, decide from the data how much each matters, and mix them. This notebook runs one causal head by hand, counts the two loss floors that show why a bigram is stuck, then trains a tiny one-head transformer on a few thousand characters of TinyStories. The full writeup is in the [lesson README](https://github.com/tamnd/soroban/tree/main/lessons/0007-attention).

![one causal head over three tokens]({ASSETS}/pattern.png)"""))

cells.append(md(
f"""## 0. One attention head over three tokens

Give three tokens the fixed embeddings `x0 = (1, 0)`, `x1 = (0, 1)`, `x2 = (1, 1)`, and set the query, key, and value projections to the identity so every number is forced. The score of position j for position i is the dot product `x[i].x[j]` divided by `sqrt(d)`, the allowed scores go through a [softmax]({MATHS}/softmax.md), and the output is the weighted sum of the earlier embeddings. Causal means position i attends only to positions at or before it. The full mechanism is on the [attention page]({MATHS}/attention.md)."""))

cells.append(code(
"""import numpy as np, math

E = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

def softmax(scores):
    s = scores - scores.max()
    e = np.exp(s)
    return e / e.sum()

def head(embeddings):
    n, d = embeddings.shape
    scale = 1.0 / math.sqrt(d)
    weights, out = [], []
    for i in range(n):
        s = np.array([embeddings[i] @ embeddings[j] for j in range(i + 1)]) * scale
        w = softmax(s)
        o = sum(w[j] * embeddings[j] for j in range(i + 1))
        weights.append(w); out.append(o)
    return weights, out

weights, out = head(E)
for i in range(3):
    print(f"tok{i}  weights", np.round(weights[i], 6), " out", np.round(out[i], 6))
assert np.allclose(weights[2], [0.248255, 0.248255, 0.503490], atol=1e-6)
assert np.allclose(out[2], [0.751745, 0.751745], atol=1e-6)"""))

cells.append(md(
f"""## 1. Why one letter is not enough: the two floors

Take the corpus aba, cbc, wrapped in the [boundary token]({MATHS}/notation.md). After b comes a in one word and c in the other, so a bigram keyed on b guesses 50/50 and pays `log 2 = 0.693147` at every position: that is the best a bigram can do. A model that could see the letter two back would be uncertain only at the first letter of each word, for a loss of `log(2)/4 = 0.173287`. The gap is the value of the context a bigram throws away."""))

cells.append(code(
"""def wrap(w): return "." + w + "."

def bigram_floor(corpus):
    nxt = {}
    for w in corpus:
        s = wrap(w)
        for a, b in zip(s, s[1:]):
            nxt.setdefault(a, {}).setdefault(b, 0); nxt[a][b] += 1
    tot, n = 0.0, 0
    for w in corpus:
        s = wrap(w)
        for a, b in zip(s, s[1:]):
            row = nxt[a]; tot += -math.log(row[b] / sum(row.values())); n += 1
    return tot / n

def context_floor(corpus):
    cont = {}
    for w in corpus:
        s = wrap(w)
        for k in range(1, len(s)):
            cont.setdefault(s[:k], {}).setdefault(s[k], 0); cont[s[:k]][s[k]] += 1
    tot, n = 0.0, 0
    for w in corpus:
        s = wrap(w)
        for k in range(1, len(s)):
            row = cont[s[:k]]; tot += -math.log(row[s[k]] / sum(row.values())); n += 1
    return tot / n

bg, cx = bigram_floor(["aba", "cbc"]), context_floor(["aba", "cbc"])
print("bigram floor ", round(bg, 6), " = log2   ", round(math.log(2), 6))
print("context floor", round(cx, 6), " = log2/4 ", round(math.log(2)/4, 6))
assert abs(bg - math.log(2)) < 1e-12 and abs(cx - math.log(2)/4) < 1e-12"""))

cells.append(md(
"""## 2. Exit test: what the scale does

Recompute token 2's weights with no scale, dividing the raw dot products `1, 1, 2` by 1 instead of `sqrt(2)`. The softmax comes out sharper, `0.211942, 0.211942, 0.576117`: more mass on the strongest match, less on the ties. The `1/sqrt(d)` scale exists to hold that sharpening back, so that at large d the softmax does not collapse to one-hot and starve the gradient."""))

cells.append(code(
"""raw = np.array([1.0, 1.0, 2.0])   # unscaled dot products for token 2
unscaled = softmax(raw)
print("unscaled weights:", np.round(unscaled, 6))
print("scaled weights:  ", np.round(weights[2], 6))
assert np.allclose(unscaled, [0.211942, 0.211942, 0.576117], atol=1e-6)"""))

cells.append(md(
"""## 3. The experiment: a few thousand characters of TinyStories

The training text is 19 short children's stories, lowercased and reduced to letters, spaces, and a little punctuation: 9311 characters, an alphabet of 33. First the baseline. A bigram counted on this exact text scores an average cross-entropy of 2.1651, the number the attention model has to beat."""))

cells.append(code(
f'''CORPUS = """{train.CORPUS}"""

chars = sorted(set(CORPUS)); V = len(chars)
stoi = {{c: i for i, c in enumerate(chars)}}
itos = {{i: c for c, i in stoi.items()}}
ids = [stoi[c] for c in CORPUS]
print(f"{{CORPUS.count(chr(10)) + 1}} stories, {{len(CORPUS)}} chars, vocab {{V}}")

N = np.zeros((V, V))
for a, b in zip(ids, ids[1:]): N[a, b] += 1
P = N / np.clip(N.sum(1, keepdims=True), 1, None)
baseline = np.mean([-math.log(max(P[a, b], 1e-12)) for a, b in zip(ids, ids[1:])])
print("bigram baseline:", round(baseline, 4))
assert 2.0 < baseline < 2.3'''))

cells.append(md(
"""## 4. A tiny transformer, trained

The smallest model that still deserves the name: a token embedding, a position embedding, one attention head, a small MLP, and a linear head, about 58 thousand weights. Train it for three thousand steps, about a minute on a CPU. Its loss starts above the bigram and settles far below, around 0.38, because it can use context the bigram cannot see. (This cell needs torch, which Colab has preinstalled.)"""))

cells.append(code(
"""try:
    import torch, torch.nn.functional as F

    torch.manual_seed(1337)
    data = torch.tensor(ids, dtype=torch.long)
    B, T, D = 32, 64, 64
    n = len(data)

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
                torch.nn.Linear(D, 4 * D), torch.nn.GELU(), torch.nn.Linear(4 * D, D))
            self.lnf = torch.nn.LayerNorm(D); self.head = torch.nn.Linear(D, V)
            self.register_buffer("mask", torch.tril(torch.ones(T, T)))

        def forward(self, idx):
            Tt = idx.shape[1]
            x = self.tok(idx) + self.pos(torch.arange(Tt))
            h = self.ln1(x)
            att = (self.q(h) @ self.k(h).transpose(-2, -1)) / math.sqrt(D)
            att = att.masked_fill(self.mask[:Tt, :Tt] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            x = x + self.proj(att @ self.v(h))
            x = x + self.mlp(self.ln2(x))
            return self.head(self.lnf(x))

    model = Block()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    nparams = sum(p.numel() for p in model.parameters())
    for _ in range(3000):
        ix = torch.randint(0, n - T - 1, (B,))
        x = torch.stack([data[i:i+T] for i in ix])
        y = torch.stack([data[i+1:i+T+1] for i in ix])
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"{nparams} params, 3000 steps: train loss {loss.item():.4f}  (bigram was {baseline:.4f})")
    assert loss.item() < 1.0

    @torch.no_grad()
    def sample(nchars=220):
        ctx = [stoi["\\n"]]; out = []
        for _ in range(nchars):
            idx = torch.tensor([ctx[-T:]])
            p = F.softmax(model(idx)[0, -1], dim=-1)
            nx = torch.multinomial(p, 1).item()
            out.append(itos[nx]); ctx.append(nx)
        return "".join(out).replace("\\n", " ")

    print("\\nsample:", sample())
except ImportError:
    print("torch not installed, skipping the training cell (Colab has it preinstalled)")"""))

cells.append(md(
"""## 5. The head, cross-checked in torch

The same three-token head built with `masked_fill` and `softmax` gives token 2 the same output, `(0.751745, 0.751745)`: the arithmetic is the mechanism, not a numpy accident."""))

cells.append(code(
"""try:
    import torch

    Et = torch.tensor(E)
    scale = 1.0 / math.sqrt(Et.shape[1])
    scores = (Et @ Et.T) * scale
    mask = torch.tril(torch.ones(3, 3))
    scores = scores.masked_fill(mask == 0, float("-inf"))
    w = torch.softmax(scores, dim=1)
    o = w @ Et
    print("torch tok2 out:", np.round(o[2].numpy(), 6))
    assert torch.allclose(o[2], torch.tensor([0.751745, 0.751745], dtype=torch.float64), atol=1e-6)
    print("torch agrees with the by-hand head")
except ImportError:
    print("torch not installed, skipping (Colab has it preinstalled)")"""))

cells.append(md(
"""## Exercises

1. Which of the three tokens has the sharpest attention, the one concentrating the most weight on a single position? Predict, then read the numbers.
2. The bigram floor is `log 2` and the context floor is `log(2)/4`. Explain in one sentence why the ratio is exactly four, using the count of positions and how many are ambiguous.
3. The training loss lands far below the bigram, but on nine thousand characters with sixty thousand weights the model mostly memorizes. What single number, not computed here, would tell you how much it actually generalizes?

Worked answers are in the [lesson README](https://github.com/tamnd/soroban/tree/main/lessons/0007-attention) and asserted in `train.py`. Lesson 0008 turns the training loop itself into an instrument, measuring exactly the generalization gap exercise 3 asks about."""))

nb.cells = cells
nbf.write(nb, str(Path(__file__).parent / "lesson.ipynb"))
print("written")
