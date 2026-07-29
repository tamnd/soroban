#!/usr/bin/env python3
"""Lesson 0004: autograd from scratch.

Builds nothing but the Value engine in micrograd.py, then proves it reproduces
lessons 0001 and 0002 to the digit with no hand-derived gradient anywhere. The
first seven printed lines are the headline table the Go runner matches byte for
byte; the rest is the nudge check, the deliberate failure, and the exit test.

    uv run train.py            # or plain: python3 train.py
    uv run train.py --torch    # cross-check the same graph against torch
"""

import argparse
import math

from micrograd import Value, mean


def one_point_graph():
    """The four-box graph of one 0001 point, walked forward and backward."""
    w, x, b, y = Value(0.0), Value(2.0), Value(0.0), Value(5.0)
    wx = w * x
    z = wx + b
    e = z - y
    L = e.sq()
    L.backward()
    return w, x, b, y, wx, z, e, L


def loss_0001(w, b, xs, ys):
    return mean([((w * xi + b) - yi).sq() for xi, yi in zip(xs, ys)])


def run_0001(lr=0.05, steps=200, check=True):
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [3.0, 5.0, 7.0, 9.0]
    w, b = 0.0, 0.0
    step1 = None
    for step in range(1, steps + 1):
        W, B = Value(w), Value(b)  # fresh leaves each step: no stale gradients
        L = loss_0001(W, B, xs, ys)
        L.backward()
        if step == 1:
            step1 = (L.data, W.grad, B.grad)
        if check:
            if step == 1:
                assert L.data == 41.0 and W.grad == -35.0 and B.grad == -12.0
            if step == 2:
                assert abs(L.data - 1.12875) < 1e-9
            if step == 3:
                assert abs(L.data - 0.043271875) < 1e-9
        w -= lr * W.grad
        b -= lr * B.grad
    return step1, (w, b)


def loss_0002(p, xs, ys):
    errs = []
    for xi, yi in zip(xs, ys):
        h1 = (p["w1"] * xi + p["b1"]).relu()
        h2 = (p["w2"] * xi + p["b2"]).relu()
        yhat = p["v1"] * h1 + p["v2"] * h2 + p["c"]
        errs.append((yhat - yi).sq())
    return mean(errs)


def run_0002(lr=0.1, steps=300, check=True):
    xs = [-2.0, -1.0, 1.0, 2.0]
    ys = [2.0, 1.0, 1.0, 2.0]
    p = {"w1": 1.0, "b1": -0.5, "w2": -1.0, "b2": -0.5, "v1": 1.0, "v2": 1.0, "c": 0.0}
    order = ("w1", "b1", "w2", "b2", "v1", "v2", "c")
    step1 = None
    for step in range(1, steps + 1):
        P = {k: Value(v) for k, v in p.items()}
        L = loss_0002(P, xs, ys)
        L.backward()
        if step == 1:
            step1 = (L.data, tuple(P[k].grad for k in order))
        if check:
            if step == 1:
                assert L.data == 0.25
                assert step1[1] == (-0.75, -0.5, 0.75, -0.5, -0.5, -0.5, -1.0)
            if step == 2:
                assert abs(L.data - 0.03631953125) < 1e-9
            if step == 3:
                assert abs(L.data - 0.01158249795541518) < 1e-9
        for k in order:
            p[k] -= lr * P[k].grad
    return step1, p


def headline():
    """The seven lines the Go runner reproduces byte for byte."""
    w, x, b, y, wx, z, e, L = one_point_graph()
    print("one point x=2 y=5 w=0 b=0")
    print(f"forward   wx {wx.data:.6f}  z {z.data:.6f}  e {e.data:.6f}  L {L.data:.6f}")
    print(
        f"backward  dL/de {e.grad:.6f}  dL/dz {z.grad:.6f}"
        f"  dL/db {b.grad:.6f}  dL/dw {w.grad:.6f}"
    )
    assert (wx.data, z.data, e.data, L.data) == (0.0, 0.0, -5.0, 25.0)
    assert (e.grad, z.grad, b.grad, w.grad) == (-10.0, -10.0, -10.0, -20.0)

    (l1, dw1, db1), (wf, bf) = run_0001()
    print(f"0001      step 1 loss {l1:.6f}  dw {dw1:.6f}  db {db1:.6f}")
    print(f"0001      final w {wf:.6f}  b {bf:.6f}")

    (l2, g), p = run_0002()
    print(
        f"0002      step 1 loss {l2:.6f}  grads "
        + " ".join(f"{v:.6f}" for v in g)
    )
    print(
        f"0002      final w1 {p['w1']:.6f} b1 {p['b1']:.6f} w2 {p['w2']:.6f}"
        f" b2 {p['b2']:.6f} v1 {p['v1']:.6f} v2 {p['v2']:.6f} c {p['c']:.6f}"
    )


def nudge_check():
    """Lesson 0001's slope-by-experiment, now checked against the engine."""
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [3.0, 5.0, 7.0, 9.0]

    def loss_at(wv, bv):
        return sum((wv * xi + bv - yi) ** 2 for xi, yi in zip(xs, ys)) / 4

    h = 1e-6
    num_dw = (loss_at(h, 0) - loss_at(0, 0)) / h
    num_db = (loss_at(0, h) - loss_at(0, 0)) / h
    print(f"nudge     dw ~ {num_dw:.4f} (engine -35)   db ~ {num_db:.4f} (engine -12)")
    assert abs(num_dw + 35) < 1e-2 and abs(num_db + 12) < 1e-2


def accumulation_check():
    """A reused node needs slopes to add: x + x is 2, x * x is 2x."""
    x = Value(3.0)
    (x + x).backward()
    print(f"reused    x+x slope {x.grad:.1f} (must be 2)")
    assert x.grad == 2.0
    x = Value(3.0)
    (x * x).backward()
    print(f"reused    x*x slope {x.grad:.1f} (must be 6)")
    assert x.grad == 6.0


def zero_grad_failure():
    """Reuse the leaves without zeroing their grad, and the descent thrashes."""
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [3.0, 5.0, 7.0, 9.0]
    w, b = Value(0.0), Value(0.0)
    lr = 0.05
    seq = []
    for step in range(1, 6):
        L = loss_0001(w, b, xs, ys)
        L.backward()  # gradients pile onto last step's, never cleared
        seq.append(L.data)
        if step == 2:
            assert abs(w.grad + 40.75) < 1e-9  # stale -35 plus correct -5.75
        w.data -= lr * w.grad
        b.data -= lr * b.grad
    print("no-zero   loss " + "  ".join(f"{v:.4f}" for v in seq))
    assert seq[2] > seq[1]  # step 3 loss climbs: the thrash, not a descent


def exit_test():
    """Add exp/log/div (done in micrograd.py) and reproduce a 0003 point."""
    z = [Value(0.0), Value(0.0), Value(0.0)]  # three scores at zero init
    exps = [zi.exp() for zi in z]
    total = exps[0] + exps[1] + exps[2]
    p = [ei / total for ei in exps]
    loss = Value(0.0) - p[1].log()  # cross-entropy, true class is 1
    loss.backward()
    grads = tuple(zi.grad for zi in z)
    print(f"0003      loss {loss.data:.6f} (ln 3)  score grads "
          + " ".join(f"{gv:.6f}" for gv in grads))
    assert abs(loss.data - math.log(3)) < 1e-12
    assert abs(grads[0] - 1 / 3) < 1e-12
    assert abs(grads[1] + 2 / 3) < 1e-12  # p - y: 1/3 - 1
    assert abs(grads[2] - 1 / 3) < 1e-12


def torch_check():
    import torch

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    y = torch.tensor([3.0, 5.0, 7.0, 9.0], dtype=torch.float64)
    w = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    loss = ((w * x + b - y) ** 2).mean()
    loss.backward()
    print(f"torch     loss {loss.item()}  dw {w.grad.item()}  db {b.grad.item()}")
    assert loss.item() == 41.0 and w.grad.item() == -35.0 and b.grad.item() == -12.0
    print("torch     agrees with the engine on the same graph")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--torch", action="store_true", help="cross-check against torch")
    args = ap.parse_args()

    headline()
    nudge_check()
    accumulation_check()
    zero_grad_failure()
    exit_test()
    if args.torch:
        torch_check()
