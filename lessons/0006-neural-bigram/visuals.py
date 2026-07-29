"""Figures for lesson 0006, rendered with manim.

The committed images in assets/ came from these scenes, so readers never need
manim installed. To regenerate:

    uv run --with manim manim -qm --format=gif -o descent.gif visuals.py Descent
    uv run --with manim manim -qh -s -o curves.png visuals.py Curves

then copy the outputs from media/ into assets/.
"""

import numpy as np
from manim import (
    BLACK,
    BLUE,
    BLUE_E,
    DOWN,
    LEFT,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Axes,
    Create,
    FadeIn,
    Scene,
    Rectangle,
    Text,
    VGroup,
    interpolate_color,
)

CORPUS = ["cat", "cot", "cab"]
ALPHABET = [".", "a", "b", "c", "o", "t"]
IDX = {c: i for i, c in enumerate(ALPHABET)}
V = len(ALPHABET)


def pairs():
    xs, ys = [], []
    for w in CORPUS:
        s = "." + w + "."
        for a, b in zip(s, s[1:]):
            xs.append(IDX[a])
            ys.append(IDX[b])
    return np.array(xs), np.array(ys)


def softmax_rows(W):
    z = W - W.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def grad(W, xs, ys):
    P = softmax_rows(W)
    G = np.zeros((V, V))
    for i in range(len(xs)):
        gi = P[xs[i]].copy()
        gi[ys[i]] -= 1
        G[xs[i]] += gi
    return G / len(xs)


def weights_at(step, xs, ys, lr=10.0):
    W = np.zeros((V, V))
    for _ in range(step):
        W = W - lr * grad(W, xs, ys)
    return W


def losses(lr, steps, xs, ys):
    W = np.zeros((V, V))
    out = []
    for _ in range(steps):
        W = W - lr * grad(W, xs, ys)
        P = softmax_rows(W)
        out.append(float(np.mean([-np.log(P[xs[i], ys[i]]) for i in range(len(xs))])))
    return out


CELL = 0.9


def grid(P):
    """A 6x6 grid of cells shaded by probability, with row and column labels."""
    g = VGroup()
    for i in range(V):
        for j in range(V):
            p = P[i, j]
            color = interpolate_color(BLACK, BLUE_E, 0.15 + 0.85 * p)
            rect = Rectangle(
                width=CELL, height=CELL, fill_color=color, fill_opacity=1.0,
                stroke_color=WHITE, stroke_width=1.5,
            ).move_to([j * CELL, -i * CELL, 0])
            g.add(rect)
            if p > 0.02:
                g.add(Text(f"{p:.2f}", font_size=18, color=WHITE).move_to(rect.get_center()))
    for i, r in enumerate(ALPHABET):
        g.add(Text(r, font_size=22, color=YELLOW).move_to([-CELL, -i * CELL, 0]))
    for j, c in enumerate(ALPHABET):
        g.add(Text(c, font_size=22, color=YELLOW).move_to([j * CELL, CELL, 0]))
    g.move_to([0, -0.2, 0])
    return g


class Descent(Scene):
    """The softmax of the weight matrix, at four training checkpoints, walking from
    a uniform gray wash toward the sharp lesson 0005 count table."""

    def construct(self):
        xs, ys = pairs()
        title = Text("gradient descent walks the softmax to the count table", font_size=26).to_edge(UP)
        self.add(title)
        checkpoints = [0, 3, 15, 200]
        for k, step in enumerate(checkpoints):
            P = softmax_rows(weights_at(step, xs, ys))
            loss = float(np.mean([-np.log(P[xs[i], ys[i]]) for i in range(len(xs))]))
            g = grid(P)
            cap = Text(f"step {step}    loss {loss:.4f}", font_size=24, color=WHITE).to_edge(DOWN)
            if k == 0:
                self.add(g, cap)
                self.wait(1.0)
                prev_g, prev_cap = g, cap
            else:
                self.play(FadeIn(g), FadeIn(cap), run_time=0.6)
                self.remove(prev_g, prev_cap)
                self.wait(1.2)
                prev_g, prev_cap = g, cap
        self.wait(1.5)


class Curves(Scene):
    """Two learning rates on the same axes: lr 10 descends smoothly, lr 50
    oscillates on a period-three sawtooth and never converges."""

    def construct(self):
        xs, ys = pairs()
        n = 50
        l10 = losses(10.0, n, xs, ys)
        l50 = losses(50.0, n, xs, ys)
        ax = Axes(
            x_range=[0, n, 10], y_range=[0, 1.4, 0.2],
            x_length=9, y_length=5.2,
            axis_config={"color": WHITE},
        )
        xlab = Text("step (0 to 50)", font_size=22).next_to(ax.x_axis, DOWN, buff=0.3)
        ylab = Text("loss (0 to 1.4)", font_size=22).rotate(np.pi / 2).next_to(ax.y_axis, LEFT, buff=0.3)
        zero = Text("0", font_size=20, color=WHITE).next_to(ax.c2p(0, 0), DOWN + LEFT, buff=0.15)
        ytop = Text("1.4", font_size=20, color=WHITE).next_to(ax.c2p(0, 1.4), LEFT, buff=0.15)
        xend = Text("50", font_size=20, color=WHITE).next_to(ax.c2p(50, 0), DOWN, buff=0.15)
        p10 = ax.plot_line_graph(
            list(range(1, n + 1)), l10, line_color=BLUE, add_vertex_dots=False, stroke_width=4,
        )
        p50 = ax.plot_line_graph(
            list(range(1, n + 1)), l50, line_color=RED, add_vertex_dots=False, stroke_width=4,
        )
        k10 = Text("lr 10: descends", font_size=22, color=BLUE).to_corner(UP + RIGHT).shift(DOWN * 0.6 + LEFT * 0.4)
        k50 = Text("lr 50: oscillates", font_size=22, color=RED).next_to(k10, DOWN, buff=0.2)
        title = Text("same model, two learning rates", font_size=26).to_edge(UP)
        self.add(ax, xlab, ylab, zero, ytop, xend, p10, p50, k10, k50, title)
