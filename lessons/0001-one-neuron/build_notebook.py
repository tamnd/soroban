"""Builds lesson.ipynb from source.

The notebook is generated, not hand-edited, so its markdown, math, and asserts
stay in sync with train.py and the README. Regenerate and re-execute with:

    uv run --with nbformat lessons/0001-one-neuron/build_notebook.py
    uv run --extra notebook python -c "import nbformat, nbclient; \
nb = nbformat.read('lessons/0001-one-neuron/lesson.ipynb', as_version=4); \
nbclient.NotebookClient(nb, timeout=120).execute(); \
nbformat.write(nb, 'lessons/0001-one-neuron/lesson.ipynb')"
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

ASSETS = "https://raw.githubusercontent.com/tamnd/soroban/main/lessons/0001-one-neuron/assets"

cells.append(md(
f"""# Lesson 0001: one neuron learns a line

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tamnd/soroban/blob/main/lessons/0001-one-neuron/lesson.ipynb)

This notebook is the runnable half of lesson 0001. The full writeup, with every step of the hand arithmetic, is in the [lesson README](https://github.com/tamnd/soroban/tree/main/lessons/0001-one-neuron). The idea here is simple: you have already computed a training run on paper, and now the code has to agree with your paper, digit for digit. Every important number below is guarded by an `assert`, so if the notebook runs top to bottom without complaint, the machine has confirmed your arithmetic.

We train the smallest possible model, $\\hat{{y}} = wx + b$, to discover the rule hiding behind four data points. The rule is $y = 2x + 1$, but the model never gets told that. It only gets the data and a way to measure its own wrongness, the [mean squared error](https://github.com/tamnd/soroban/blob/main/maths/mean-squared-error.md):

$$L = \\frac{{1}}{{N}}\\sum_{{i=1}}^{{N}} (\\hat{{y}}_i - y_i)^2$$

In words: guess minus truth for each of the $N$ points, square each one, average them. If any formula in this notebook stalls you, the repo's [maths shelf](https://github.com/tamnd/soroban/tree/main/maths) has one plain-language page per symbol and idea, worked out on this lesson's numbers.

Here is the whole notebook in one animation, rendered with manim from this lesson's `visuals.py`:

![the line learning]({ASSETS}/fit.gif)"""))

cells.append(md(
"""## 0. Setup

On Google Colab there is nothing to install, numpy, torch and matplotlib all come preinstalled, just run the cells. Locally you need python with numpy and matplotlib (torch is optional, one cell uses it and will skip itself if missing). The zero-effort way is [uv](https://docs.astral.sh/uv/): `uv run --with jupyter,numpy,matplotlib jupyter lab lesson.ipynb`."""))

cells.append(code(
"""import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0])  # inputs
y = np.array([3.0, 5.0, 7.0, 9.0])  # true answers, secretly y = 2x + 1

w, b = 0.0, 0.0  # the two knobs of the model, starting fully ignorant

y_hat = w * x + b
e = y_hat - y
loss = (e**2).mean()

print("predictions:", y_hat)
print("errors:     ", e)
print("loss:       ", loss)
assert loss == 41.0  # the number you got on paper"""))

cells.append(md(
f"""## 1. Which way should w move? Ask, by nudging

Before any theory, do the dumbest possible experiment: change $w$ by a tiny amount, recompute the loss, and see which way it moved. The ratio of loss-change to knob-change is the slope, and its sign tells you which direction is downhill. That ratio, taken to the limit of an infinitely small nudge, is exactly what the [derivative](https://github.com/tamnd/soroban/blob/main/maths/derivative.md) notation means:

$$\\frac{{\\partial L}}{{\\partial w}} = \\lim_{{h \\to 0}} \\frac{{L(w+h) - L(w)}}{{h}}$$

In a picture: hold $b$ at 0 and the loss traces a bowl as $w$ varies. We are about to measure the steepness of that bowl at our starting point, and the answer will be about $-35$.

![the loss bowl over w]({ASSETS}/slope.png)"""))

cells.append(code(
"""loss_nudged = (((w + 0.001) * x + b - y) ** 2).mean()
slope_w = (loss_nudged - loss) / 0.001
print("measured slope for w:", slope_w)  # about -35, so pushing w up pushes the loss down"""))

cells.append(md(
"""## 2. The same slope from the chain rule, no nudging needed

Each data point contributes $e^2$ to the loss, where $e = wx + b - y$. The slope of $e^2$ with respect to $e$ is $2e$, and the slope of $e$ with respect to $w$ is $x$, so the slope of that point's loss with respect to $w$ is $2ex$. Multiplying the stage slopes like this is the [chain rule](https://github.com/tamnd/soroban/blob/main/maths/chain-rule.md), and averaging over the points gives the gradient of the full loss. Same story for $b$, whose stage slope is 1:

$$\\frac{\\partial L}{\\partial w} = \\frac{2}{N}\\sum_{i=1}^{N} e_i x_i \\qquad \\frac{\\partial L}{\\partial b} = \\frac{2}{N}\\sum_{i=1}^{N} e_i$$"""))

cells.append(code(
"""dw = 2 * (e * x).mean()
db = 2 * e.mean()
print("dL/dw =", dw, "  dL/db =", db)

assert dw == -35.0 and db == -12.0  # your paper again
print("formula", dw, "vs nudge", slope_w, ": the nudge is off by the nudge size, the formula is exact")"""))

cells.append(md(
"""## 3. The training loop

[Gradient descent](https://github.com/tamnd/soroban/blob/main/maths/gradient-descent.md) is one move applied forever: step each knob a little against its slope,

$$w \\leftarrow w - \\mathrm{lr}\\cdot\\frac{\\partial L}{\\partial w} \\qquad b \\leftarrow b - \\mathrm{lr}\\cdot\\frac{\\partial L}{\\partial b}$$

The step size $\\mathrm{lr}$ is the [learning rate](https://github.com/tamnd/soroban/blob/main/maths/gradient-descent.md), here 0.05. The asserts pin the first three losses to the hand run: 41, then 1.12875, then 0.043271875."""))

cells.append(code(
"""w, b, lr = 0.0, 0.0, 0.05
losses = []

for step in range(1, 201):
    y_hat = w * x + b
    e = y_hat - y
    loss = (e**2).mean()
    dw = 2 * (e * x).mean()
    db = 2 * e.mean()
    losses.append(loss)

    if step == 1:
        assert loss == 41.0 and dw == -35.0 and db == -12.0
    if step == 2:
        assert abs(loss - 1.12875) < 1e-9
    if step == 3:
        assert abs(loss - 0.043271875) < 1e-9

    if step in (1, 2, 3, 10, 50, 200):
        print(f"step {step:3d}  loss {loss:.9f}  w {w:.6f}  b {b:.6f}")

    w -= lr * dw
    b -= lr * db

print(f"final     w {w:.6f}  b {b:.6f}  (hidden truth: w = 2, b = 1)")"""))

cells.append(md(
f"""Numbers first, picture second. The curve below is the same story on a log scale: a huge early drop while $w$ races into place, then a long patient tail while $b$ catches up (the loss is far less sensitive to $b$, so its steps are smaller). The journey of the pair $(w, b)$ across the loss landscape makes the asymmetry vivid, one diagonal sprint for $w$, then a slow crawl up the valley for $b$:

![the path of (w, b) across the loss landscape]({ASSETS}/trajectory.png)"""))

cells.append(code(
"""import matplotlib.pyplot as plt

plt.figure(figsize=(7, 3))
plt.semilogy(range(1, 201), losses)
plt.xlabel("step")
plt.ylabel("loss (log scale)")
plt.title("one neuron learning a line")
plt.tight_layout()
plt.show()"""))

cells.append(md(
"""## 4. Torch computes the same gradients without being told the formula

We derived `2*e*x` on paper. [Automatic differentiation](https://github.com/tamnd/soroban/blob/main/maths/autodiff.md) is that derivation, mechanized: torch records every operation used to build the loss, then applies the chain rule backward through the recording. Nobody types the gradient formula anywhere, and out come the exact same -35 and -12."""))

cells.append(code(
"""try:
    import torch

    xt = torch.tensor([1.0, 2.0, 3.0, 4.0])
    yt = torch.tensor([3.0, 5.0, 7.0, 9.0])
    wt = torch.zeros(1, requires_grad=True)
    bt = torch.zeros(1, requires_grad=True)

    tloss = ((wt * xt + bt - yt) ** 2).mean()
    tloss.backward()

    print("torch:", tloss.item(), wt.grad.item(), bt.grad.item())
    assert tloss.item() == 41.0
    assert wt.grad.item() == -35.0 and bt.grad.item() == -12.0
    print("torch agrees with your paper")
except ImportError:
    print("torch not installed, skipping this check (on Colab it just runs)")"""))

cells.append(md(
"""## 5. Break it on purpose

The learning rate is a bet about how far the slope stays honest. Push it and watch the two classic failure signatures: at lr = 0.1 the loss overshoots the valley floor every step and oscillates its way down anyway, and at lr = 0.2 every overshoot lands higher than it started and the loss runs away to infinity. Keep these shapes; they are what broken training looks like at every scale."""))

cells.append(code(
"""for lr_try in (0.05, 0.1, 0.2):
    w, b = 0.0, 0.0
    trace = []
    for step in range(5):
        e = w * x + b - y
        trace.append((e**2).mean())
        w -= lr_try * 2 * (e * x).mean()
        b -= lr_try * 2 * e.mean()
    print(f"lr {lr_try:4}: " + "  ".join(f"{l:10.2f}" for l in trace))"""))

cells.append(md(
"""## 6. Exercises

1. Delete the factor 2 from both gradient formulas and rerun the loop. Predict what happens before you run it.
2. Delete `b` from the model entirely, so `y_hat = w * x`, and run 400 steps. The loss stops falling at a floor. Compute the floor exactly (it is a nice fraction).
3. Make the data noisy, `y = [3.1, 4.8, 7.2, 8.9]`, and rerun. Where does the loss end up, and is that a failure?

Answers and discussion are at the end of the [lesson README](https://github.com/tamnd/soroban/tree/main/lessons/0001-one-neuron), along with the exit test: a fresh dataset to train by hand, with the numbers to check yourself against. When your paper and your asserts agree, you are done, and lesson 0002 adds a hidden layer."""))

nb.cells = cells
nbf.write(nb, str(Path(__file__).parent / "lesson.ipynb"))
print("written")
