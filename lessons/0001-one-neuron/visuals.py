"""Figures for lesson 0001, rendered with manim.

The committed images in assets/ came from these scenes, so readers never
need manim installed. To regenerate:

    uv run --with manim manim -qm --format=gif -o fit.gif visuals.py LineLearns
    uv run --with manim manim -qh -s -o slope.png visuals.py SlopeAtStart
    uv run --with manim manim -qh -s -o trajectory.png visuals.py KnobTrajectory

then copy the outputs from media/ into assets/.
"""

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREY,
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
    VMobject,
)

X = np.array([1.0, 2.0, 3.0, 4.0])
Y = np.array([3.0, 5.0, 7.0, 9.0])


def trajectory(steps=200, lr=0.05):
    """Replay the lesson's training run and record (w, b, loss) per step."""
    w, b = 0.0, 0.0
    out = [(w, b, ((w * X + b - Y) ** 2).mean())]
    for _ in range(steps):
        e = w * X + b - Y
        w -= lr * 2 * (e * X).mean()
        b -= lr * 2 * e.mean()
        out.append((w, b, ((w * X + b - Y) ** 2).mean()))
    return out


class LineLearns(Scene):
    """The model's line snapping onto the data as training runs."""

    def construct(self):
        ax = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 10, 2],
            x_length=7,
            y_length=5,
            axis_config={"include_numbers": True, "label_constructor": Text},
        )
        points = VGroup(*[Dot(ax.c2p(x, y), color=YELLOW) for x, y in zip(X, Y)])

        traj = trajectory()
        w, b, loss = traj[0]
        line = ax.plot(lambda t: w * t + b, x_range=[0, 5], color=BLUE)

        def labels(step, loss):
            step_t = Text(f"step {step}", font_size=30).to_corner(UR)
            loss_t = Text(f"loss {loss:.3f}", font_size=30, color=ORANGE)
            loss_t.next_to(step_t, DOWN, aligned_edge=RIGHT)
            return VGroup(step_t, loss_t)

        info = labels(0, loss)
        self.add(ax, points, line, info)
        self.wait(0.8)

        for step in [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200]:
            w, b, loss = traj[step]
            new_line = ax.plot(lambda t, w=w, b=b: w * t + b, x_range=[0, 5], color=BLUE)
            self.play(
                Transform(line, new_line),
                Transform(info, labels(step, loss)),
                run_time=0.55,
            )
        self.wait(1.5)


class SlopeAtStart(Scene):
    """The loss as a function of w alone (b fixed at 0), with the slope at
    the starting point. The bowl is L(w) = 7.5 w^2 - 35 w + 41 and the
    tangent at w = 0 has the famous slope of -35."""

    def construct(self):
        ax = Axes(
            x_range=[-1, 6, 1],
            y_range=[0, 90, 20],
            x_length=8,
            y_length=5,
            axis_config={"include_numbers": True, "label_constructor": Text},
        )
        bowl = ax.plot(lambda w: 7.5 * w * w - 35 * w + 41, x_range=[-1, 5.7], color=BLUE)
        start = Dot(ax.c2p(0, 41), color=YELLOW)
        tangent = ax.plot(lambda w: 41 - 35 * w, x_range=[-0.6, 1.0], color=ORANGE)

        slope_t = Text("slope here = -35", font_size=28, color=ORANGE)
        slope_t.next_to(ax.c2p(1.0, 6), RIGHT)
        start_t = Text("start: w = 0, loss = 41", font_size=28)
        start_t.next_to(start, UP).shift(RIGHT * 1.6)
        axis_t = Text("loss as w varies (b held at 0)", font_size=26, color=GREY)
        axis_t.to_corner(UR)

        self.add(ax, bowl, tangent, start, slope_t, start_t, axis_t)


class KnobTrajectory(Scene):
    """The path of (w, b) during training, over level curves of the loss.
    Shows w racing to 2 and overshooting while b crawls toward 1."""

    def construct(self):
        ax = Axes(
            x_range=[0, 2.6, 0.5],
            y_range=[0, 1.4, 0.5],
            x_length=8,
            y_length=5,
            axis_config={"include_numbers": True, "label_constructor": Text, "font_size": 24},
        )

        def loss(w, b):
            return ((w * X + b - Y) ** 2).mean()

        # Level curves of the loss, drawn by scanning angles around the minimum.
        contours = VGroup()
        for level in (0.02, 0.2, 1.0, 4.0, 12.0, 30.0):
            pts = []
            for theta in np.linspace(0, 2 * np.pi, 240):
                lo, hi = 0.0, 4.0
                for _ in range(40):
                    mid = (lo + hi) / 2
                    w = 2 + mid * np.cos(theta)
                    b = 1 + mid * np.sin(theta)
                    if loss(w, b) < level:
                        lo = mid
                    else:
                        hi = mid
                w = 2 + lo * np.cos(theta)
                b = 1 + lo * np.sin(theta)
                if 0 <= w <= 2.6 and 0 <= b <= 1.4:
                    pts.append(ax.c2p(w, b))
                else:
                    if len(pts) > 2:
                        contours.add(VMobject(color=GREY, stroke_width=1.5).set_points_smoothly(pts))
                    pts = []
            if len(pts) > 2:
                contours.add(VMobject(color=GREY, stroke_width=1.5).set_points_smoothly(pts))

        traj = trajectory()
        path = VMobject(color=BLUE, stroke_width=3)
        path.set_points_as_corners([ax.c2p(w, b) for w, b, _ in traj])
        dots = VGroup(*[Dot(ax.c2p(w, b), radius=0.05, color=YELLOW) for w, b, _ in traj[:4]])

        start_t = Text("start (0, 0)", font_size=26).next_to(ax.c2p(0, 0), UP + RIGHT, buff=0.1)
        end_t = Text("truth (2, 1)", font_size=26, color=ORANGE)
        end_t.next_to(ax.c2p(2, 1), DOWN, buff=0.15)
        truth = Dot(ax.c2p(2, 1), color=ORANGE)
        w_axis = Text("w", font_size=28).next_to(ax.x_axis, DOWN, buff=0.2)
        b_axis = Text("b", font_size=28).next_to(ax.y_axis, UP, buff=0.1)

        self.add(ax, contours, path, dots, truth, start_t, end_t, w_axis, b_axis)
