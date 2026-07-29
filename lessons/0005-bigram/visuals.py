"""Figures for lesson 0005, rendered with manim.

The committed images in assets/ came from these scenes, so readers never need
manim installed. To regenerate:

    uv run --with manim manim -qh -s -o matrix.png visuals.py Matrix
    uv run --with manim manim -qm --format=gif -o walk.gif visuals.py Walk

then copy the outputs from media/ into assets/.
"""

import numpy as np
from manim import (
    BLACK,
    BLUE_E,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Create,
    FadeIn,
    Rectangle,
    Scene,
    SurroundingRectangle,
    Text,
    VGroup,
    interpolate_color,
)

CORPUS = ["cat", "cot", "cab"]
ALPHABET = ['.', 'a', 'b', 'c', 'o', 't']
IDX = {c: i for i, c in enumerate(ALPHABET)}
V = len(ALPHABET)


def tables():
    N = np.zeros((V, V), dtype=int)
    for w in CORPUS:
        s = '.' + w + '.'
        for a, b in zip(s, s[1:]):
            N[IDX[a], IDX[b]] += 1
    rows = N.sum(1, keepdims=True)
    P = np.divide(N, rows, out=np.zeros((V, V)), where=rows > 0)
    return N, P


CELL = 0.9


def grid(P):
    """A 6x6 grid of cells shaded by probability, with row and column labels."""
    g = VGroup()
    cells = {}
    for i in range(V):
        for j in range(V):
            p = P[i, j]
            color = interpolate_color(BLACK, BLUE_E, 0.15 + 0.85 * p) if p > 0 else BLACK
            rect = Rectangle(
                width=CELL, height=CELL, fill_color=color, fill_opacity=1.0,
                stroke_color=WHITE, stroke_width=1.5,
            ).move_to([j * CELL, -i * CELL, 0])
            g.add(rect)
            cells[(i, j)] = rect
            if p > 0:
                g.add(Text(f"{p:.2f}", font_size=18, color=WHITE).move_to(rect.get_center()))
    # labels
    for i, r in enumerate(ALPHABET):
        g.add(Text(r, font_size=22, color=YELLOW).move_to([-CELL, -i * CELL, 0]))
    for j, c in enumerate(ALPHABET):
        g.add(Text(c, font_size=22, color=YELLOW).move_to([j * CELL, CELL, 0]))
    g.move_to([0, 0, 0])
    return g, cells


class Matrix(Scene):
    """The bigram table as a heatmap: rows are the current letter, columns the
    next, brightness the probability. Blank (black) means a pair never seen."""

    def construct(self):
        _, P = tables()
        g, _ = grid(P)
        g.shift(DOWN * 0.4)
        title = Text("P(next | current): the whole model is this table", font_size=26)
        title.to_edge(UP)
        rowlab = Text("current", font_size=20, color=YELLOW).next_to(g, LEFT, buff=0.6)
        collab = Text("next", font_size=20, color=YELLOW).next_to(g, UP, buff=0.5)
        self.add(g, title, rowlab, collab)


class Walk(Scene):
    """Sampling 'cat': start at the boundary row, light the picked cell, move to
    that letter's row, repeat until the boundary is drawn again."""

    def construct(self):
        _, P = tables()
        g, cells = grid(P)
        self.add(g)
        title = Text("sampling: walk the table until it draws the boundary", font_size=24).to_edge(UP)
        self.add(title)

        # (current, next) picks that spell .cat.
        steps = [('.', 'c'), ('c', 'a'), ('a', 't'), ('t', '.')]
        word = Text("word: ", font_size=28, color=WHITE).to_edge(DOWN)
        self.add(word)
        built = ""
        for a, b in steps:
            box = SurroundingRectangle(cells[(IDX[a], IDX[b])], color=YELLOW, buff=0)
            self.play(Create(box), run_time=0.7)
            self.wait(0.2)
            if b != '.':
                built += b
                new = Text("word: " + built, font_size=28, color=YELLOW).to_edge(DOWN)
                self.play(FadeIn(new), run_time=0.3)
                self.remove(word)
                word = new
            self.wait(0.3)
        done = Text("word: " + built + "  (boundary drawn, stop)", font_size=28, color=YELLOW).to_edge(DOWN)
        self.remove(word)
        self.add(done)
        self.wait(1.5)
