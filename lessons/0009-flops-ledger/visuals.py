"""Figures for lesson 0009, rendered with manim.

The committed images in assets/ came from these scenes, so readers never need
manim installed. To regenerate:

    uv run --with manim manim -qh -s -o ledger.png visuals.py Ledger
    uv run --with manim manim -qh -s -o roofline.png visuals.py Roofline

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
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Rectangle,
    Scene,
    Text,
    VGroup,
)

# The measured 4090 numbers from train.py --bench, eager/compiled/batch-24.
CONFIGS = ["eager\nbatch 12", "compiled\nbatch 12", "eager\nbatch 24"]
TFLOPS = [92.1, 110.8, 38.0]
MFU = [55.8, 67.1, 23.0]
BAR_COLOR = [BLUE, GREEN, RED]
PEAK = 165.2


class Ledger(Scene):
    """The paper ledger as a chain: one matmul, to 6N per token, to 6ND for the
    run, to a floor in days once a GPU's peak rate is divided in."""

    def construct(self):
        title = Text("the FLOPs ledger: from one matmul to days", font_size=30).to_edge(UP)
        lines = [
            ("one matmul (m x k)(k x n)", "2 m k n flops", WHITE),
            ("forward one token", "2N flops", WHITE),
            ("backward one token", "4N flops", WHITE),
            ("forward + backward", "6N = 741,920,256 per token", BLUE),
            ("times 300 billion tokens", "6ND = 2.23e20 flops", YELLOW),
            ("divide by 4090 peak 165.2 TFLOPs at 100% mfu", "floor 15.6 days, never met", RED),
        ]
        rows = VGroup()
        for left, right, color in lines:
            l = Text(left, font_size=24)
            r = Text(right, font_size=24, color=color)
            r.next_to(l, RIGHT, buff=0.4)
            rows.add(VGroup(l, r))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        rows.next_to(title, DOWN, buff=0.6)
        arrows = VGroup()
        for a, b in zip(rows[:-1], rows[1:]):
            arrows.add(Arrow(a.get_left() + DOWN * 0.25, b.get_left() + UP * 0.25,
                             buff=0.05, color=GREY, stroke_width=3, max_tip_length_to_length_ratio=0.15))
        self.add(title, rows, arrows)


class Roofline(Scene):
    """Achieved throughput of the three configs against the 4090's advertised
    165.2 TFLOPs peak. The dashed peak is the roof; each bar's height is its
    MFU, and batch 24 is the cliff."""

    def construct(self):
        title = Text("measured TFLOPs against the 4090's 165.2 peak", font_size=28).to_edge(UP)
        base_y = -2.6
        top_y = 2.4
        span = top_y - base_y
        peak_y = base_y + span * (PEAK / 180.0)   # scale so 180 TFLOPs is the top
        baseline = Text("0", font_size=18).move_to([-5.3, base_y, 0])
        peak_line = Rectangle(width=9.2, height=0.02, color=GREY, fill_opacity=1).move_to([-0.6, peak_y, 0])
        peak_lab = Text("165.2 TFLOPs peak (MFU = 1)", font_size=20, color=GREY).next_to(peak_line, UP, buff=0.1)
        bars = VGroup()
        labels = VGroup()
        xs = [-3.4, -0.6, 2.2]
        for x, cfg, tf, mfu, color in zip(xs, CONFIGS, TFLOPS, MFU, BAR_COLOR):
            h = span * (tf / 180.0)
            bar = Rectangle(width=1.6, height=h, color=color, fill_opacity=0.85, stroke_width=0)
            bar.move_to([x, base_y + h / 2, 0])
            bars.add(bar)
            val = Text(f"{tf:.0f} TFLOPs\n{mfu:.0f}% MFU", font_size=20, color=color)
            val.next_to(bar, UP, buff=0.12)
            name = Text(cfg, font_size=20).next_to(bar, DOWN, buff=0.15)
            labels.add(val, name)
        self.add(title, peak_line, peak_lab, baseline, bars, labels)
