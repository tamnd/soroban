"""Figures for lesson 0003, rendered with manim.

The committed images in assets/ came from these scenes, so readers never
need manim installed. To regenerate:

    uv run --with manim manim -qm --format=gif -o fit.gif visuals.py CurvesLearn
    uv run --with manim manim -qh -s -o price.png visuals.py PriceCurve
    uv run --with manim manim -qh -s -o curves.png visuals.py TrainedCurves

then copy the outputs from media/ into assets/.
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREEN,
    ORANGE,
    RIGHT,
    UP,
    UR,
    YELLOW,
    Axes,
    Dot,
    Scene,
    Text,
    Transform,
    VGroup,
)

X = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
LABELS = np.array([0, 0, 1, 1, 2, 2])
CLASS_COLORS = [BLUE, GREEN, ORANGE]
CLASS_NAMES = ["ice", "water", "steam"]


def probs(w, b, x):
    z = np.outer(np.atleast_1d(x), w) + b
    ez = np.exp(z - z.max(axis=1, keepdims=True))
    return ez / ez.sum(axis=1, keepdims=True)


def trajectory(steps=300, lr=0.1):
    """Replay the lesson's training run and record (w, b, loss) per step."""
    Y = np.eye(3)[LABELS]
    w, b = np.zeros(3), np.zeros(3)
    out = []
    for _ in range(steps + 1):
        P = probs(w, b, X)
        loss = -np.log(P[np.arange(len(X)), LABELS]).mean()
        out.append((w.copy(), b.copy(), loss))
        dZ = (P - Y) / len(X)
        w = w - lr * (dZ * X[:, None]).sum(axis=0)
        b = b - lr * dZ.sum(axis=0)
    return out


def prob_axes():
    return Axes(
        x_range=[-3, 3, 1],
        y_range=[0, 1.05, 0.5],
        x_length=8,
        y_length=5,
        axis_config={"include_numbers": True, "label_constructor": Text},
    )


def data_dots(ax):
    return VGroup(*[
        Dot(ax.c2p(x, 0), color=CLASS_COLORS[k]) for x, k in zip(X, LABELS)
    ])


def prob_curves(ax, w, b):
    return VGroup(*[
        ax.plot(
            lambda t, k=k: float(probs(w, b, t)[0, k]),
            x_range=[-3, 3, 0.02],
            color=CLASS_COLORS[k],
            use_smoothing=False,
        )
        for k in range(3)
    ])


class CurvesLearn(Scene):
    """The three class probability curves rising out of the flat one-third
    start and carving the line into three territories."""

    def construct(self):
        ax = prob_axes()
        traj = trajectory()

        w, b, loss = traj[0]
        curves = prob_curves(ax, w, b)

        def labels(step, loss):
            step_t = Text(f"step {step}", font_size=30).to_corner(UR)
            loss_t = Text(f"loss {loss:.3f}", font_size=30, color=YELLOW)
            loss_t.next_to(step_t, DOWN, aligned_edge=RIGHT)
            return VGroup(step_t, loss_t)

        legend = VGroup(*[
            Text(name, font_size=26, color=CLASS_COLORS[k])
            for k, name in enumerate(CLASS_NAMES)
        ])
        legend.arrange(RIGHT, buff=0.5).to_edge(UP)

        info = labels(0, loss)
        self.add(ax, data_dots(ax), curves, legend, info)
        self.wait(0.8)

        for step in [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 300]:
            w, b, loss = traj[step]
            self.play(
                Transform(curves, prob_curves(ax, w, b)),
                Transform(info, labels(step, loss)),
                run_time=0.55,
            )
        self.wait(1.5)


class PriceCurve(Scene):
    """Cross-entropy's price list: -ln p, nearly free when confident and
    right, unbounded when confident and wrong."""

    def construct(self):
        ax = Axes(
            x_range=[0, 1.05, 0.25],
            y_range=[0, 5, 1],
            x_length=8,
            y_length=5,
            axis_config={"include_numbers": True, "label_constructor": Text},
        )
        curve = ax.plot(
            lambda p: -np.log(p),
            x_range=[0.008, 1.04, 0.002],
            color=YELLOW,
            use_smoothing=False,
        )

        marks = [
            (1.0, 0.0, "p = 1 costs 0: right and certain, free", (0.62, 2.2)),
            (1 / 3, np.log(3), "p = 1/3 costs 1.099: knows nothing", (0.66, 1.1)),
            (0.01, -np.log(0.01), "p = 0.01 costs 4.605: confident and wrong", (0.42, 4.6)),
        ]
        dots = VGroup()
        for p, cost, text, at in marks:
            d = Dot(ax.c2p(p, cost), color=ORANGE)
            t = Text(text, font_size=24, color=ORANGE).move_to(ax.c2p(*at))
            dots.add(VGroup(d, t))

        title = Text("the price of a probability: -ln p", font_size=28, color=YELLOW)
        title.to_corner(UR)
        self.add(ax, curve, dots, title)


class TrainedCurves(Scene):
    """The trained probability curves over the six data points: confident
    at the ends, honestly torn at the boundaries."""

    def construct(self):
        ax = prob_axes()
        traj = trajectory()
        w, b, _ = traj[300]
        curves = prob_curves(ax, w, b)

        names = VGroup(
            Text("ice", font_size=28, color=BLUE).next_to(ax.c2p(-2.5, 0.93), UP, buff=0.1),
            Text("water", font_size=28, color=GREEN).next_to(ax.c2p(0, 0.72), UP, buff=0.15),
            Text("steam", font_size=28, color=ORANGE).next_to(ax.c2p(2.5, 0.93), UP, buff=0.1),
        )
        note = Text("torn at the borders, certain past the data", font_size=26)
        note.to_edge(UP)

        self.add(ax, data_dots(ax), curves, names, note)
