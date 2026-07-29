#!/usr/bin/env python3
"""Lesson 0002: a hidden layer learns a V.

Reproduces the hand computation from the lesson README, asserting every
number along the way, then trains to convergence. Runs anywhere python and
numpy exist:

    uv run train.py             # or plain: python3 train.py
    uv run train.py --lr 0.5    # watch both relu neurons die
    uv run train.py --torch     # check all seven gradients against torch
"""

import argparse

import numpy as np


def relu(z):
    return np.maximum(0.0, z)


def train(lr=0.1, steps=300, check=True):
    x = np.array([-2.0, -1.0, 1.0, 2.0])
    y = np.array([2.0, 1.0, 1.0, 2.0])

    # The fixed init from the lesson: each hidden neuron owns one arm of
    # the V, at the wrong height. Real nets initialize randomly; here the
    # numbers are chosen so paper and machine can agree digit for digit.
    w1, b1 = 1.0, -0.5
    w2, b2 = -1.0, -0.5
    v1, v2, c = 1.0, 1.0, 0.0

    for step in range(1, steps + 1):
        z1, z2 = w1 * x + b1, w2 * x + b2
        h1, h2 = relu(z1), relu(z2)
        g1, g2 = (z1 > 0).astype(float), (z2 > 0).astype(float)
        e = v1 * h1 + v2 * h2 + c - y
        loss = (e**2).mean()

        dc = 2 * e.mean()
        dv1 = 2 * (e * h1).mean()
        dv2 = 2 * (e * h2).mean()
        dw1 = 2 * (e * v1 * g1 * x).mean()
        db1 = 2 * (e * v1 * g1).mean()
        dw2 = 2 * (e * v2 * g2 * x).mean()
        db2 = 2 * (e * v2 * g2).mean()

        # Step 1 is all halves and quarters, binary-exact, so it must match
        # the hand run exactly. Later steps involve 0.1, which has no exact
        # binary representation, hence the 1e-9 tolerance.
        if check:
            if step == 1:
                assert loss == 0.25
                assert (dw1, db1, dw2, db2) == (-0.75, -0.5, 0.75, -0.5)
                assert (dv1, dv2, dc) == (-0.5, -0.5, -1.0)
            if step == 2:
                assert abs(loss - 0.03631953125) < 1e-9
            if step == 3:
                assert abs(loss - 0.01158249795541518) < 1e-9

        if step in (1, 2, 3, 10, 50) or step == steps:
            print(f"step {step:3d}  loss {loss:.9f}")

        w1 -= lr * dw1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2
        v1 -= lr * dv1
        v2 -= lr * dv2
        c -= lr * dc

    print(
        f"final     w1 {w1:.6f}  b1 {b1:.6f}  w2 {w2:.6f}  b2 {b2:.6f}"
        f"  v1 {v1:.6f}  v2 {v2:.6f}  c {c:.6f}"
    )
    return w1, b1, w2, b2, v1, v2, c


def torch_check():
    import torch

    x = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    y = torch.tensor([2.0, 1.0, 1.0, 2.0])
    params = {
        "w1": 1.0, "b1": -0.5, "w2": -1.0, "b2": -0.5,
        "v1": 1.0, "v2": 1.0, "c": 0.0,
    }
    p = {k: torch.tensor([v], requires_grad=True) for k, v in params.items()}

    y_hat = (
        p["v1"] * torch.relu(p["w1"] * x + p["b1"])
        + p["v2"] * torch.relu(p["w2"] * x + p["b2"])
        + p["c"]
    )
    loss = ((y_hat - y) ** 2).mean()
    loss.backward()

    grads = {k: v.grad.item() for k, v in p.items()}
    print(f"torch: loss {loss.item()}  grads {grads}")
    assert loss.item() == 0.25
    expected = {
        "w1": -0.75, "b1": -0.5, "w2": 0.75, "b2": -0.5,
        "v1": -0.5, "v2": -0.5, "c": -1.0,
    }
    assert grads == expected
    print("torch agrees with the hand arithmetic, all seven gradients exact")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lr", type=float, default=0.1, help="learning rate")
    p.add_argument("--steps", type=int, default=300, help="training steps")
    p.add_argument("--torch", action="store_true", help="run the torch autograd check")
    args = p.parse_args()

    # The asserts encode the lr=0.1 hand run, so they only apply there.
    train(lr=args.lr, steps=args.steps, check=(args.lr == 0.1))
    if args.torch:
        torch_check()
