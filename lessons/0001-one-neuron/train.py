#!/usr/bin/env python3
"""Lesson 0001: one neuron learns a line.

Reproduces the hand computation from the lesson README, asserting every
number along the way, then trains to convergence. Runs anywhere python and
numpy exist:

    uv run train.py            # or plain: python3 train.py
    uv run train.py --lr 0.2   # watch it explode
    uv run train.py --torch    # check the gradients against torch.autograd
"""

import argparse

import numpy as np


def train(lr=0.05, steps=200, check=True):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([3.0, 5.0, 7.0, 9.0])
    w, b = 0.0, 0.0

    for step in range(1, steps + 1):
        y_hat = w * x + b
        e = y_hat - y
        loss = (e**2).mean()
        dw = 2 * (e * x).mean()
        db = 2 * e.mean()

        # Step 1 is pure integer arithmetic, so it must match the hand run
        # exactly. Later steps involve 0.05, which has no exact binary
        # representation, hence the 1e-9 tolerance.
        if check:
            if step == 1:
                assert loss == 41.0 and dw == -35.0 and db == -12.0
            if step == 2:
                assert abs(loss - 1.12875) < 1e-9
                assert abs(dw + 5.75) < 1e-9 and abs(db + 2.05) < 1e-9
            if step == 3:
                assert abs(loss - 0.043271875) < 1e-9

        if step in (1, 2, 3, 10, 50) or step == steps:
            print(f"step {step:3d}  loss {loss:.9f}  w {w:.6f}  b {b:.6f}")

        w -= lr * dw
        b -= lr * db

    print(f"final     w {w:.6f}  b {b:.6f}  (hidden truth: w = 2, b = 1)")
    return w, b


def torch_check():
    import torch

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = torch.tensor([3.0, 5.0, 7.0, 9.0])
    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    loss = ((w * x + b - y) ** 2).mean()
    loss.backward()

    print(f"torch: loss {loss.item()}  dL/dw {w.grad.item()}  dL/db {b.grad.item()}")
    assert loss.item() == 41.0
    assert w.grad.item() == -35.0 and b.grad.item() == -12.0
    print("torch agrees with the hand arithmetic")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lr", type=float, default=0.05, help="learning rate")
    p.add_argument("--steps", type=int, default=200, help="training steps")
    p.add_argument("--torch", action="store_true", help="run the torch autograd check")
    args = p.parse_args()

    # The asserts encode the lr=0.05 hand run, so they only apply there.
    train(lr=args.lr, steps=args.steps, check=(args.lr == 0.05))
    if args.torch:
        torch_check()
