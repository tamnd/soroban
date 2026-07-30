"""Figures for lesson 0008, rendered with manim.

The committed images in assets/ came from these scenes, so readers never need
manim installed. To regenerate:

    uv run --with manim manim -qh -s -o overfit.png visuals.py Overfit
    uv run --with manim manim -qh -s -o gap.png visuals.py Gap

then copy the outputs from media/ into assets/.
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY,
    LEFT,
    RED,
    UP,
    WHITE,
    YELLOW,
    Axes,
    DashedLine,
    Dot,
    Scene,
    Text,
    VGroup,
)

# The by-hand dataset and the two fits from overfit.go / train.py.
TRAIN_X = [0.0, 1.0, 2.0]
TRAIN_Y = [0.5, 1.3, 1.5]
VAL_X = [3.0, 4.0]
VAL_Y = [2.0, 2.5]


def line(x):
    return 0.5 * x + 0.6


def parab(x):
    return -0.3 * x ** 2 + 1.1 * x + 0.5


# The D=64 train/val curves, captured from train.py --train, seed 1337, sampled
# every 150 steps. Train keeps falling; val bottoms at step 600 then climbs.
STEPS = [0, 150, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500,
         1650, 1800, 1950, 2100, 2250, 2400, 2550, 2700, 2850, 3000]
TRAIN = [3.7251, 1.9132, 1.5505, 1.393, 1.2384, 1.1084, 0.9804, 0.8875, 0.7884,
         0.717, 0.6453, 0.5944, 0.5387, 0.4913, 0.4591, 0.4172, 0.3993, 0.3797,
         0.3663, 0.3417, 0.3232]
VAL = [3.7511, 2.1122, 1.9887, 1.917, 1.8799, 1.9346, 1.9677, 2.0983, 2.2633,
       2.4542, 2.6453, 2.7893, 2.9975, 3.0949, 3.2308, 3.4156, 3.6032, 3.8262,
       3.9806, 4.2301, 4.1363]
BIGRAM_BASELINE = 2.1651
BEST_STEP, BEST_VAL = 600, 1.8799


class Overfit(Scene):
    """The by-hand core: a line and a parabola fit to three noisy points. The
    parabola hits every training point and then dives away from the held-out ones."""

    def construct(self):
        ax = Axes(
            x_range=[-0.3, 4.5, 1], y_range=[-0.5, 3, 1],
            x_length=9, y_length=5.4,
            axis_config={"color": WHITE},
        )
        xs = np.linspace(-0.2, 4.35, 200)
        line_g = ax.plot_line_graph(xs, [line(x) for x in xs], line_color=BLUE,
                                    add_vertex_dots=False, stroke_width=4)
        parab_g = ax.plot_line_graph(xs, [parab(x) for x in xs], line_color=RED,
                                     add_vertex_dots=False, stroke_width=4)
        dots = VGroup()
        for x, y in zip(TRAIN_X, TRAIN_Y):
            dots.add(Dot(ax.c2p(x, y), color=YELLOW, radius=0.09))
        for x, y in zip(VAL_X, VAL_Y):
            dots.add(Dot(ax.c2p(x, y), color=GREEN, radius=0.09))
        line_lab = Text("line: train 0.02, held-out 0.01", font_size=22, color=BLUE).to_corner(UP + LEFT).shift(DOWN * 0.95)
        parab_lab = Text("parabola: train 0.00, held-out 3.285", font_size=22, color=RED).next_to(line_lab, DOWN, aligned_edge=LEFT, buff=0.2)
        pts = Text("yellow: 3 training points   green: 2 held-out", font_size=20).to_edge(DOWN)
        title = Text("same points, two fits: the parabola memorizes and then misses", font_size=26).to_edge(UP)
        self.add(ax, line_g, parab_g, dots, line_lab, parab_lab, pts, title)


class Gap(Scene):
    """The transformer's train and validation loss. Training loss falls forever;
    validation loss bottoms early and then climbs as the model starts memorizing."""

    def construct(self):
        ax = Axes(
            x_range=[0, 3000, 500], y_range=[0, 4.5, 1],
            x_length=9, y_length=5.2,
            axis_config={"color": WHITE},
        )
        xlab = Text("training step (0 to 3000)", font_size=22).next_to(ax.x_axis, DOWN, buff=0.3)
        ylab = Text("loss (0 to 4.5)", font_size=22).rotate(np.pi / 2).next_to(ax.y_axis, LEFT, buff=0.3)
        zero = Text("0", font_size=20).next_to(ax.c2p(0, 0), DOWN + LEFT, buff=0.15)
        xend = Text("3000", font_size=20).next_to(ax.c2p(3000, 0), DOWN, buff=0.15)
        train_g = ax.plot_line_graph(STEPS, TRAIN, line_color=BLUE, add_vertex_dots=False, stroke_width=4)
        val_g = ax.plot_line_graph(STEPS, VAL, line_color=RED, add_vertex_dots=False, stroke_width=4)
        base = DashedLine(ax.c2p(0, BIGRAM_BASELINE), ax.c2p(3000, BIGRAM_BASELINE), color=GREY, stroke_width=3)
        base_lab = Text(f"bigram baseline {BIGRAM_BASELINE}", font_size=20, color=GREY).next_to(ax.c2p(3000, BIGRAM_BASELINE), UP + LEFT, buff=0.15)
        stop = Dot(ax.c2p(BEST_STEP, BEST_VAL), color=YELLOW, radius=0.1)
        stop_lab = Text("stop here: val bottoms at 1.88", font_size=20, color=YELLOW).next_to(stop, UP, buff=0.2)
        train_lab = Text("train loss", font_size=22, color=BLUE).next_to(ax.c2p(3000, TRAIN[-1]), DOWN + LEFT, buff=0.15)
        val_lab = Text("val loss", font_size=22, color=RED).next_to(ax.c2p(3000, VAL[-1]), LEFT, buff=0.2)
        title = Text("train loss keeps falling; val loss turns back up: that gap is overfitting", font_size=24).to_edge(UP)
        self.add(ax, xlab, ylab, zero, xend, base, base_lab, train_g, val_g, stop, stop_lab, train_lab, val_lab, title)
