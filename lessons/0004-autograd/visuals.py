"""Figures for lesson 0004, rendered with manim.

The committed images in assets/ came from these scenes, so readers never need
manim installed. To regenerate:

    uv run --with manim manim -qh -s -o graph.png visuals.py Graph
    uv run --with manim manim -qm --format=gif -o backward.gif visuals.py Backward
    uv run --with manim manim -qh -s -o thrash.png visuals.py Thrash

then copy the outputs from media/ into assets/.
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREY,
    LEFT,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Arrow,
    Axes,
    Create,
    Dot,
    FadeIn,
    Circle,
    RoundedRectangle,
    Scene,
    Text,
    VGroup,
    Write,
)

# Node positions for the four-box graph of one 0001 point (x=2, y=5, w=0, b=0).
OPS = {
    "mul": (-4.2, 0.0, "*"),
    "add": (-1.4, 0.0, "+"),
    "sub": (1.4, 0.0, "-"),
    "sq": (4.2, 0.0, "( )^2"),
}
LEAVES = {
    "w": (-6.0, 1.4),
    "x": (-6.0, -1.4),
    "b": (-2.8, 1.7),
    "y": (-0.2, -1.8),
}
# Forward value carried on each wire, and the backward slope along it.
WIRES = [
    ("w", "mul", None, "dL/dw = -20"),
    ("x", "mul", None, None),
    ("mul", "add", "wx = 0", "dL/dwx = -10"),
    ("b", "add", None, "dL/db = -10"),
    ("add", "sub", "z = 0", "dL/dz = -10"),
    ("y", "sub", None, None),
    ("sub", "sq", "e = -5", "dL/de = -10"),
]


def pos(name):
    if name in OPS:
        gx, gy, _ = OPS[name]
        return np.array([gx, gy, 0.0])
    lx, ly = LEAVES[name]
    return np.array([lx, ly, 0.0])


class Graph(Scene):
    """The computation graph, forward values in white, backward slopes in
    yellow: one picture of the whole backward pass."""

    def construct(self):
        op_nodes = VGroup()
        for name, (gx, gy, sym) in OPS.items():
            circ = Circle(radius=0.55, color=BLUE, fill_opacity=0.15).move_to([gx, gy, 0])
            label = Text(sym, font_size=26, color=BLUE).move_to(circ.get_center())
            op_nodes.add(VGroup(circ, label))

        leaf_nodes = VGroup()
        for name, (lx, ly) in LEAVES.items():
            box = RoundedRectangle(
                width=0.9, height=0.6, corner_radius=0.12, color=WHITE
            ).move_to([lx, ly, 0])
            label = Text(name, font_size=26).move_to(box.get_center())
            leaf_nodes.add(VGroup(box, label))

        arrows, fwd, bwd = VGroup(), VGroup(), VGroup()
        for src, dst, fval, bval in WIRES:
            a = Arrow(pos(src), pos(dst), buff=0.6, stroke_width=3, color=GREY)
            arrows.add(a)
            mid = a.get_center()
            if fval:
                fwd.add(Text(fval, font_size=22, color=WHITE).next_to(mid, UP, buff=0.08))
            if bval:
                bwd.add(Text(bval, font_size=20, color=YELLOW).next_to(mid, DOWN, buff=0.08))

        out = Arrow(pos("sq"), [5.9, 0, 0], buff=0.6, stroke_width=3, color=GREY)
        loss = Text("L = 25", font_size=26, color=WHITE).next_to(out, RIGHT, buff=0.1)
        seed = Text("dL/dL = 1", font_size=20, color=YELLOW).next_to(out, DOWN, buff=0.08)

        title = Text("forward fills the wires, backward walks them back", font_size=24)
        title.to_edge(UP)

        self.add(arrows, out, op_nodes, leaf_nodes, fwd, bwd, loss, seed, title)


class Backward(Scene):
    """The backward walk animated: seed L with 1, then let the slope propagate
    box by box down to the leaves."""

    def construct(self):
        op_nodes = VGroup()
        for name, (gx, gy, sym) in OPS.items():
            circ = Circle(radius=0.55, color=BLUE, fill_opacity=0.15).move_to([gx, gy, 0])
            label = Text(sym, font_size=26, color=BLUE).move_to(circ.get_center())
            op_nodes.add(VGroup(circ, label))
        leaf_nodes = VGroup()
        for name, (lx, ly) in LEAVES.items():
            box = RoundedRectangle(
                width=0.9, height=0.6, corner_radius=0.12, color=WHITE
            ).move_to([lx, ly, 0])
            leaf_nodes.add(VGroup(box, Text(name, font_size=26).move_to(box.get_center())))
        arrows = {}
        for src, dst, _, _ in WIRES:
            arrows[(src, dst)] = Arrow(
                pos(src), pos(dst), buff=0.6, stroke_width=3, color=GREY
            )
        out = Arrow(pos("sq"), [5.9, 0, 0], buff=0.6, stroke_width=3, color=GREY)
        loss = Text("L = 25", font_size=26).next_to(out, RIGHT, buff=0.1)

        self.add(VGroup(*arrows.values()), out, op_nodes, leaf_nodes, loss)

        # Seed, then each wire's slope in the order the walk fills them.
        steps = [
            (out, "dL/dL = 1"),
            (arrows[("sub", "sq")], "dL/de = -10"),
            (arrows[("add", "sub")], "dL/dz = -10"),
            (arrows[("b", "add")], "dL/db = -10"),
            (arrows[("mul", "add")], "dL/dwx = -10"),
            (arrows[("w", "mul")], "dL/dw = -20"),
        ]
        for arrow, text in steps:
            label = Text(text, font_size=22, color=YELLOW).next_to(arrow, DOWN, buff=0.08)
            self.play(arrow.animate.set_color(YELLOW), Write(label), run_time=0.6)
            self.wait(0.25)
        self.wait(1.5)


def trajectories():
    """Two 0001 runs: the correct one that zeros gradients, and the buggy one
    that does not."""
    xs = np.array([1.0, 2.0, 3.0, 4.0])
    ys = np.array([3.0, 5.0, 7.0, 9.0])
    lr = 0.05

    def loss(w, b):
        return float(((w * xs + b - ys) ** 2).mean())

    good, bad = [], []
    w = b = 0.0
    gw = gb = 0.0  # the buggy run never resets these
    wb = bb = 0.0
    for _ in range(12):
        e = w * xs + b - ys
        good.append(loss(w, b))
        w -= lr * 2 * (e * xs).mean()
        b -= lr * 2 * e.mean()

        eb = wb * xs + bb - ys
        bad.append(loss(wb, bb))
        gw += 2 * (eb * xs).mean()  # accumulate, never zeroed
        gb += 2 * eb.mean()
        wb -= lr * gw
        bb -= lr * gb
    return good, bad


class Thrash(Scene):
    """The deliberate failure: skipping zero-grad turns a smooth descent into a
    thrash."""

    def construct(self):
        good, bad = trajectories()
        ax = Axes(
            x_range=[1, 12, 1],
            y_range=[0, 55, 10],
            x_length=9,
            y_length=5,
            axis_config={"include_numbers": True, "label_constructor": Text},
        )
        gline = ax.plot_line_graph(
            range(1, 13), good, line_color=BLUE, add_vertex_dots=True, vertex_dot_radius=0.05
        )
        bline = ax.plot_line_graph(
            range(1, 13), bad, line_color=YELLOW, add_vertex_dots=True, vertex_dot_radius=0.05
        )
        gl = Text("zeroed each step: descends", font_size=24, color=BLUE).to_corner(UP + LEFT)
        bl = Text("never zeroed: thrashes", font_size=24, color=YELLOW).next_to(gl, DOWN, aligned_edge=LEFT)
        self.add(ax, gline, bline, gl, bl)
