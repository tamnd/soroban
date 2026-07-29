"""Figures for lesson 0002, rendered with manim.

The committed images in assets/ came from these scenes, so readers never
need manim installed. To regenerate:

    uv run --with manim manim -qm --format=gif -o fit.gif visuals.py VeeLearns
    uv run --with manim manim -qh -s -o floor.png visuals.py LineFloor
    uv run --with manim manim -qh -s -o arms.png visuals.py TwoArms

then copy the outputs from media/ into assets/.
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY,
    ORANGE,
    RIGHT,
    UP,
    UR,
    YELLOW,
    Axes,
    DashedLine,
    Dot,
    Scene,
    Text,
    Transform,
    VGroup,
)

X = np.array([-2.0, -1.0, 1.0, 2.0])
Y = np.array([2.0, 1.0, 1.0, 2.0])


def relu(z):
    return np.maximum(0.0, z)


def trajectory(steps=300, lr=0.1):
    """Replay the lesson's training run and record (params, loss) per step."""
    p = np.array([1.0, -0.5, -1.0, -0.5, 1.0, 1.0, 0.0])

    def predict(p, xs):
        w1, b1, w2, b2, v1, v2, c = p
        return v1 * relu(w1 * xs + b1) + v2 * relu(w2 * xs + b2) + c

    out = [(p.copy(), ((predict(p, X) - Y) ** 2).mean())]
    for _ in range(steps):
        w1, b1, w2, b2, v1, v2, c = p
        z1, z2 = w1 * X + b1, w2 * X + b2
        h1, h2 = relu(z1), relu(z2)
        g1, g2 = (z1 > 0).astype(float), (z2 > 0).astype(float)
        e = v1 * h1 + v2 * h2 + c - Y
        g = np.array([
            2 * (e * v1 * g1 * X).mean(),
            2 * (e * v1 * g1).mean(),
            2 * (e * v2 * g2 * X).mean(),
            2 * (e * v2 * g2).mean(),
            2 * (e * h1).mean(),
            2 * (e * h2).mean(),
            2 * e.mean(),
        ])
        p = p - lr * g
        out.append((p.copy(), ((predict(p, X) - Y) ** 2).mean()))
    return out


def predict(p, xs):
    w1, b1, w2, b2, v1, v2, c = p
    return v1 * relu(w1 * xs + b1) + v2 * relu(w2 * xs + b2) + c


class VeeLearns(Scene):
    """The model's bent line finding the V as training runs."""

    def construct(self):
        ax = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 3, 1],
            x_length=8,
            y_length=5,
            axis_config={"include_numbers": True, "label_constructor": Text},
        )
        points = VGroup(*[Dot(ax.c2p(x, y), color=YELLOW) for x, y in zip(X, Y)])

        traj = trajectory()
        p, loss = traj[0]
        curve = ax.plot(
            lambda t, p=p: float(predict(p, np.array([t]))[0]),
            x_range=[-3, 3, 0.02],
            color=BLUE,
            use_smoothing=False,
        )

        def labels(step, loss):
            step_t = Text(f"step {step}", font_size=30).to_corner(UR)
            loss_t = Text(f"loss {loss:.3f}", font_size=30, color=ORANGE)
            loss_t.next_to(step_t, DOWN, aligned_edge=RIGHT)
            return VGroup(step_t, loss_t)

        info = labels(0, loss)
        self.add(ax, points, curve, info)
        self.wait(0.8)

        for step in [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 300]:
            p, loss = traj[step]
            new_curve = ax.plot(
                lambda t, p=p: float(predict(p, np.array([t]))[0]),
                x_range=[-3, 3, 0.02],
                color=BLUE,
                use_smoothing=False,
            )
            self.play(
                Transform(curve, new_curve),
                Transform(info, labels(step, loss)),
                run_time=0.55,
            )
        self.wait(1.5)


class LineFloor(Scene):
    """The V data and the best line that exists, stuck at loss 0.25. By
    symmetry the best slope is 0, so the best line is the best constant,
    y = 1.5, and every point misses by exactly 0.5."""

    def construct(self):
        ax = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 3, 1],
            x_length=8,
            y_length=5,
            axis_config={"include_numbers": True, "label_constructor": Text},
        )
        points = VGroup(*[Dot(ax.c2p(x, y), color=YELLOW) for x, y in zip(X, Y)])
        truth = ax.plot(lambda t: abs(t), x_range=[-3, 3, 0.02], color=GREY, use_smoothing=False)
        best = ax.plot(lambda t: 1.5, x_range=[-3, 3], color=BLUE)
        misses = VGroup(*[
            DashedLine(ax.c2p(x, y), ax.c2p(x, 1.5), color=ORANGE) for x, y in zip(X, Y)
        ])

        best_t = Text("best line: y = 1.5, loss 0.25", font_size=28, color=BLUE)
        best_t.to_corner(UR)
        miss_t = Text("every point missed by 0.5", font_size=26, color=ORANGE)
        miss_t.next_to(best_t, DOWN, aligned_edge=RIGHT)
        truth_t = Text("the V the data came from", font_size=26, color=GREY)
        truth_t.next_to(ax.c2p(1.4, 2.6), RIGHT, buff=0.1)

        self.add(ax, truth, best, misses, points, best_t, miss_t, truth_t)


class TwoArms(Scene):
    """The identity that beats the line: relu(x) + relu(-x) = |x|. Each
    relu contributes one arm and stays flat on the other side."""

    def construct(self):
        ax = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 3, 1],
            x_length=8,
            y_length=5,
            axis_config={"include_numbers": True, "label_constructor": Text},
        )
        right_arm = ax.plot(lambda t: relu(t), x_range=[-3, 3, 0.02], color=GREEN, use_smoothing=False)
        left_arm = ax.plot(lambda t: relu(-t), x_range=[-3, 3, 0.02], color=ORANGE, use_smoothing=False)
        vee = ax.plot(lambda t: abs(t) + 0.06, x_range=[-3, 3, 0.02], color=YELLOW, use_smoothing=False)

        r_t = Text("relu(x): the right arm", font_size=26, color=GREEN)
        r_t.next_to(ax.c2p(2.0, 1.2), RIGHT, buff=0.1).shift(DOWN * 0.6)
        l_t = Text("relu(-x): the left arm", font_size=26, color=ORANGE)
        l_t.next_to(ax.c2p(-2.0, 1.2), UP, buff=0.1).shift(DOWN * 1.4 + RIGHT * 0.4)
        s_t = Text("their sum: |x|, the V exactly", font_size=28, color=YELLOW)
        s_t.to_corner(UR)

        self.add(ax, right_arm, left_arm, vee, r_t, l_t, s_t)
