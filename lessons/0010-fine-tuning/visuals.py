"""Figures for lesson 0010, rendered with manim.

The committed images in assets/ came from these scenes, so readers never need
manim installed. To regenerate:

    uv run --with manim manim -qh -s -o factor.png visuals.py Factor
    uv run --with manim manim -qh -s -o memory.png visuals.py Memory

then copy the outputs from media/ into assets/.
"""

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
    Rectangle,
    Scene,
    Text,
    VGroup,
)


class Factor(Scene):
    """The full 4096-by-4096 update against the two thin strips B and A that
    stand in for it: a big square of 16,777,216 numbers replaced by 131,072."""

    def construct(self):
        title = Text("one weight update: full square versus B times A", font_size=30).to_edge(UP)

        # The full update: a large square.
        full = Rectangle(width=2.6, height=2.6, color=BLUE, fill_opacity=0.25, stroke_width=2)
        full.move_to([-3.6, -0.3, 0])
        full_lab = Text("full update\n4096 x 4096\n16,777,216", font_size=20, color=BLUE)
        full_lab.next_to(full, DOWN, buff=0.2)

        eq = Text("has the same effect as", font_size=22, color=GREY).move_to([-0.4, -0.3, 0])

        # B is a tall thin strip, A is a wide short strip.
        b = Rectangle(width=0.28, height=2.6, color=GREEN, fill_opacity=0.6, stroke_width=1)
        b.move_to([1.9, -0.3, 0])
        b_lab = Text("B\n4096 x 16", font_size=18, color=GREEN).next_to(b, DOWN, buff=0.2)
        times = Text("x", font_size=26, color=WHITE).next_to(b, RIGHT, buff=0.25)
        a = Rectangle(width=2.6, height=0.28, color=YELLOW, fill_opacity=0.6, stroke_width=1)
        a.next_to(times, RIGHT, buff=0.25)
        a_lab = Text("A\n16 x 4096", font_size=18, color=YELLOW).next_to(a, DOWN, buff=0.5)

        count = Text("131,072 numbers, 0.78% of the square", font_size=22, color=GREEN)
        count.to_edge(DOWN, buff=0.35)

        self.add(title, full, full_lab, eq, b, b_lab, times, a, a_lab, count)


class Memory(Scene):
    """The memory ledger as bars against the 24 GB line of a 4090: a full
    fine-tune towers over it at 107.8 GB, QLoRA fits under it at about 4 GB."""

    def construct(self):
        title = Text("fine-tuning Llama-2-7B: memory against a 24 GB 4090", font_size=28).to_edge(UP)
        base_y = -2.7
        top = 108.0  # GB at the top of the axis

        def h(gb):
            return 4.8 * (gb / top)

        # The 24 GB card ceiling as a dashed line.
        ceil_y = base_y + h(24)
        ceiling = Rectangle(width=9.0, height=0.03, color=RED, fill_opacity=1).move_to([0, ceil_y, 0])
        ceil_lab = Text("24 GB, one 4090", font_size=20, color=RED).next_to(ceiling, RIGHT, buff=0.1).shift(LEFT * 0.2 + UP * 0.2)

        bars = [
            ("full fine-tune\n107.8 GB", 107.8, RED, -3.0),
            ("QLoRA base\n3.37 GB", 3.37, GREEN, 0.3),
            ("QLoRA total\n~4 GB", 4.0, BLUE, 3.0),
        ]
        group = VGroup()
        for label, gb, color, x in bars:
            bar = Rectangle(width=1.7, height=h(gb), color=color, fill_opacity=0.8, stroke_width=0)
            bar.move_to([x, base_y + h(gb) / 2, 0])
            lab = Text(label, font_size=19, color=color).next_to(bar, DOWN, buff=0.15)
            group.add(bar, lab)

        note = Text("full fits nothing consumer; QLoRA fits with room", font_size=20, color=GREY)
        note.to_edge(DOWN, buff=0.3)
        self.add(title, ceiling, ceil_lab, group, note)
