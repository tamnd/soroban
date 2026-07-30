"""Figures for lesson 0007, rendered with manim.

The committed images in assets/ came from these scenes, so readers never need
manim installed. To regenerate:

    uv run --with manim manim -qh -s -o pattern.png visuals.py Pattern
    uv run --with manim manim -qh -s -o descent.png visuals.py Descent

then copy the outputs from media/ into assets/.
"""

import math

import numpy as np
from manim import (
    BLACK,
    BLUE,
    BLUE_E,
    DOWN,
    GREY_E,
    LEFT,
    RED,
    UP,
    WHITE,
    YELLOW,
    Axes,
    DashedLine,
    Rectangle,
    Scene,
    Text,
    VGroup,
    interpolate_color,
)

# The three-token embeddings of the by-hand head; dot products come out whole.
E = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def head_weights(embeddings):
    """The causal attention weights, one ragged row per query position."""
    n, d = embeddings.shape
    scale = 1.0 / math.sqrt(d)
    rows = []
    for i in range(n):
        s = np.array([embeddings[i] @ embeddings[j] for j in range(i + 1)]) * scale
        s = s - s.max()
        e = np.exp(s)
        rows.append(e / e.sum())
    return rows


# The transformer's training loss on the TinyStories text, seed 1337, sampled
# every 150 steps; captured from a run of train.py --train so the curve is real.
STEPS = [0, 150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500,
         1650, 1800, 1950, 2100, 2250, 2400, 2550, 2700, 2850, 3000]
LOSS = [3.7442, 1.9299, 1.575, 1.3817, 1.2042, 1.1406, 0.9771, 0.9682, 0.8819,
        0.7751, 0.6815, 0.6154, 0.5537, 0.5396, 0.5267, 0.4693, 0.4486, 0.4446,
        0.433, 0.3726, 0.3832]
BIGRAM_BASELINE = 2.1651

CELL = 1.3


class Pattern(Scene):
    """One causal head over three tokens: a lower-triangular grid of attention
    weights, the future masked out, each token mixing itself with the past."""

    def construct(self):
        rows = head_weights(E)
        g = VGroup()
        for i in range(3):
            for j in range(3):
                if j <= i:
                    w = rows[i][j]
                    color = interpolate_color(BLACK, BLUE_E, 0.2 + 0.8 * w)
                    rect = Rectangle(
                        width=CELL, height=CELL, fill_color=color, fill_opacity=1.0,
                        stroke_color=WHITE, stroke_width=1.5,
                    ).move_to([j * CELL, -i * CELL, 0])
                    g.add(rect)
                    g.add(Text(f"{w:.3f}", font_size=24, color=WHITE).move_to(rect.get_center()))
                else:
                    rect = Rectangle(
                        width=CELL, height=CELL, fill_color=GREY_E, fill_opacity=1.0,
                        stroke_color=WHITE, stroke_width=1.5,
                    ).move_to([j * CELL, -i * CELL, 0])
                    g.add(rect)
                    g.add(Text("masked", font_size=16, color=WHITE).move_to(rect.get_center()))
        for i in range(3):
            g.add(Text(f"tok {i}", font_size=22, color=YELLOW).move_to([-CELL, -i * CELL, 0]))
        for j in range(3):
            g.add(Text(f"key {j}", font_size=22, color=YELLOW).move_to([j * CELL, CELL, 0]))
        g.move_to([0, -0.3, 0])
        title = Text("one causal head: each token attends to itself and the past", font_size=26).to_edge(UP)
        row_note = Text("rows are query positions, columns are keys, each row sums to 1", font_size=20).to_edge(DOWN)
        self.add(g, title, row_note)


class Descent(Scene):
    """The attention model's training loss falling from above the bigram baseline
    to well below it: the value of the context a bigram cannot see."""

    def construct(self):
        ax = Axes(
            x_range=[0, 3000, 500], y_range=[0, 4, 1],
            x_length=9, y_length=5.2,
            axis_config={"color": WHITE},
        )
        xlab = Text("training step (0 to 3000)", font_size=22).next_to(ax.x_axis, DOWN, buff=0.3)
        ylab = Text("loss (0 to 4)", font_size=22).rotate(np.pi / 2).next_to(ax.y_axis, LEFT, buff=0.3)
        zero = Text("0", font_size=20).next_to(ax.c2p(0, 0), DOWN + LEFT, buff=0.15)
        ytop = Text("4", font_size=20).next_to(ax.c2p(0, 4), LEFT, buff=0.15)
        xend = Text("3000", font_size=20).next_to(ax.c2p(3000, 0), DOWN, buff=0.15)
        curve = ax.plot_line_graph(
            STEPS, LOSS, line_color=BLUE, add_vertex_dots=False, stroke_width=4,
        )
        base = DashedLine(
            ax.c2p(0, BIGRAM_BASELINE), ax.c2p(3000, BIGRAM_BASELINE),
            color=RED, stroke_width=3,
        )
        base_lab = Text(f"bigram baseline {BIGRAM_BASELINE}", font_size=22, color=RED).next_to(
            ax.c2p(3000, BIGRAM_BASELINE), UP + LEFT, buff=0.2
        )
        attn_lab = Text("attention head 0.38", font_size=22, color=BLUE).next_to(
            ax.c2p(3000, LOSS[-1]), UP + LEFT, buff=0.2
        )
        title = Text("the attention model falls below the bigram it started above", font_size=26).to_edge(UP)
        self.add(ax, xlab, ylab, zero, ytop, xend, base, base_lab, curve, attn_lab, title)
