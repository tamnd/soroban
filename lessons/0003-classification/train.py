#!/usr/bin/env python3
"""Lesson 0003: ice, water, steam, a first classifier.

Reproduces the hand computation from the lesson README, asserting every
number along the way, then trains to convergence. Runs anywhere python and
numpy exist:

    uv run train.py                     # or plain: python3 train.py
    uv run train.py --lr 25             # thrash, then land anyway
    uv run train.py --lr 1000 --naive   # watch the naive softmax die of nan
    uv run train.py --torch             # check loss and gradients against torch
"""

import argparse
import math

import numpy as np


def softmax(Z, naive=False):
    if not naive:
        Z = Z - Z.max(axis=1, keepdims=True)
    ez = np.exp(Z)
    return ez / ez.sum(axis=1, keepdims=True)


def train(lr=0.1, steps=300, check=True, naive=False):
    x = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
    labels = np.array([0, 0, 1, 1, 2, 2])  # ice, ice, water, water, steam, steam
    N = len(x)
    Y = np.eye(3)[labels]  # one-hot targets

    # Zero init, on purpose. It was fatal in 0002; here the targets tell
    # the three rows apart, and every starting probability is exactly 1/3.
    w = np.zeros(3)
    b = np.zeros(3)

    for step in range(1, steps + 1):
        Z = np.outer(x, w) + b
        P = softmax(Z, naive=naive)
        loss = -np.log(P[np.arange(N), labels]).mean()

        # The star of the lesson: dL/dz = probability minus target.
        dZ = (P - Y) / N
        dw = (dZ * x[:, None]).sum(axis=0)
        db = dZ.sum(axis=0)

        # New assert regime: 1/3 and ln 3 are not binary-exact, so unlike
        # lessons 0001 and 0002 even step 1 carries a tolerance. 1e-12 is
        # far below anything a maths mistake could produce and far above
        # float dust from summation order.
        if check:
            if step == 1:
                assert abs(loss - math.log(3)) < 1e-12
                assert np.abs(dw - [0.5, 0.0, -0.5]).max() < 1e-12
                assert np.abs(db).max() < 1e-12
            if step == 2:
                assert abs(loss - 1.0500696359446569) < 1e-9
            if step == 3:
                assert abs(loss - 1.0070093819902546) < 1e-9

        if step in (1, 2, 3, 10, 50) or step == steps:
            print(f"step {step:3d}  loss {loss:.9f}")

        w -= lr * dw
        b -= lr * db

    print(
        f"final     w ({w[0]:.6f}, {w[1]:.6f}, {w[2]:.6f})"
        f"  b ({b[0]:.6f}, {b[1]:.6f}, {b[2]:.6f})"
    )
    return w, b


def torch_check():
    import torch

    x = torch.tensor([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], dtype=torch.float64)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    w = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(3, dtype=torch.float64, requires_grad=True)

    # cross_entropy fuses softmax and the log loss, the same cancellation
    # the lesson does on paper.
    Z = x[:, None] * w + b
    loss = torch.nn.functional.cross_entropy(Z, labels)
    loss.backward()

    print(f"torch: loss {loss.item()}  dw {w.grad.tolist()}  db {b.grad.tolist()}")
    assert abs(loss.item() - math.log(3)) < 1e-12
    assert all(abs(g - e) < 1e-12 for g, e in zip(w.grad.tolist(), [0.5, 0.0, -0.5]))
    assert all(abs(g) < 1e-12 for g in b.grad.tolist())
    print("torch agrees with the hand arithmetic, loss and all six gradients")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lr", type=float, default=0.1, help="learning rate")
    p.add_argument("--steps", type=int, default=300, help="training steps")
    p.add_argument(
        "--naive", action="store_true", help="skip the subtract-max trick in softmax"
    )
    p.add_argument("--torch", action="store_true", help="run the torch autograd check")
    args = p.parse_args()

    # The asserts encode the lr=0.1 hand run, so they only apply there.
    train(lr=args.lr, steps=args.steps, check=(args.lr == 0.1), naive=args.naive)
    if args.torch:
        torch_check()
