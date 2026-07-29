"""Lesson 0006: the neural bigram learns the table lesson 0005 counted.

Same corpus as 0005 (cat, cot, cab), a different model. Instead of counting, keep
a 6x6 matrix of weights, read one row per current letter (an embedding lookup),
softmax the row into next-letter probabilities, and train the weights by
gradient descent. The cross-entropy gradient on the logits is predicted frequency
minus observed frequency, so gradient descent walks the softmax rows to the
observed frequencies, which are exactly the 0005 count table.

Every asserted number was computed by hand in the spec first. Run it with:

    uv run lessons/0006-neural-bigram/train.py
    uv run --with torch lessons/0006-neural-bigram/train.py --torch

The first seven printed lines match `go run ./cmd/soroban 0006` byte for byte.
"""

import math
import sys

import numpy as np

CORPUS = ["cat", "cot", "cab"]
ALPHABET = [".", "a", "b", "c", "o", "t"]
IDX = {c: i for i, c in enumerate(ALPHABET)}
V = len(ALPHABET)
DICE = [0.5, 0.2, 0.8, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.2, 0.3, 0.5]


def bigrams(word):
    s = "." + word + "."
    return list(zip(s, s[1:]))


def pairs():
    xs, ys = [], []
    for w in CORPUS:
        for a, b in bigrams(w):
            xs.append(IDX[a])
            ys.append(IDX[b])
    return np.array(xs), np.array(ys)


def softmax_rows(W):
    z = W - W.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def loss_of(W, xs, ys):
    P = softmax_rows(W)
    return np.mean([-math.log(P[xs[i], ys[i]]) for i in range(len(xs))])


def grad(W, xs, ys, reg=0.0):
    P = softmax_rows(W)
    G = np.zeros((V, V))
    for i in range(len(xs)):
        gi = P[xs[i]].copy()
        gi[ys[i]] -= 1
        G[xs[i]] += gi
    return G / len(xs) + reg * W


def train(xs, ys, lr, steps, reg=0.0):
    W = np.zeros((V, V))
    for _ in range(steps):
        W = W - lr * grad(W, xs, ys, reg)
    return W


def sample(W, dice, n_words=3):
    P = softmax_rows(W)
    it = iter(dice)
    words = []
    for _ in range(n_words):
        cur, out = IDX["."], []
        while True:
            u = next(it)
            cum = 0.0
            nxt = IDX["."]
            for s in range(V):
                cum += P[cur, s]
                if cum >= u:
                    nxt = s
                    break
            if nxt == IDX["."]:
                break
            out.append(ALPHABET[nxt])
            cur = nxt
        words.append("".join(out))
    return words


def headline(xs, ys):
    W0 = np.zeros((V, V))
    init_loss = loss_of(W0, xs, ys)
    assert abs(init_loss - math.log(6)) < 1e-12

    # one full-batch step, learning rate 10, watch W[c,a] leave zero
    G = grad(W0, xs, ys)
    row_c = G[IDX["c"]]
    expect = np.array([(0.5 - o) / 12 for o in (0, 2, 0, 0, 1, 0)])  # columns . a b c o t
    assert np.allclose(row_c, expect)
    assert abs(row_c[IDX["a"]] - (-0.125)) < 1e-12
    wca_before = W0[IDX["c"], IDX["a"]]
    wca_after = wca_before - 10 * G[IDX["c"], IDX["a"]]
    assert wca_after == 1.25
    W1 = W0 - 10 * G
    assert abs(loss_of(W1, xs, ys) - 0.874450) < 1e-6

    # train 200 steps at learning rate 10
    W = train(xs, ys, 10, 200)
    trained = loss_of(W, xs, ys)
    P = softmax_rows(W)
    ac, oc = P[IDX["c"], IDX["a"]], P[IDX["c"], IDX["o"]]
    ta, ba = P[IDX["a"], IDX["t"]], P[IDX["a"], IDX["b"]]
    assert abs(trained - 0.277674) < 1e-5
    assert abs(ac - 0.665334) < 1e-5 and abs(oc - 0.331997) < 1e-5
    assert abs(ta - 0.497989) < 1e-5 and abs(ba - 0.497989) < 1e-5
    assert math.log(3) / 4 < trained  # approached from above, never reached
    assert sample(W, DICE) == ["cat", "cot", "cab"]

    print("corpus cat cot cab")
    print("model  W[current] one-hot lookup = an embedding, 6x6 = 36 weights")
    print(f"init   zeros, softmax uniform 1/6, loss {init_loss:.6f}  (log 6)")
    print(f"grad   row c step 1: predicted minus observed, W[c,a] {wca_before:.6f} -> {wca_after:.6f}")
    print(f"train  200 steps lr 10: loss {init_loss:.6f} -> {trained:.6f}  (min log3/4 {math.log(3)/4:.6f})")
    print(f"learned a|c {ac:.6f}  o|c {oc:.6f}  (counts {2/3:.6f} {1/3:.6f})")
    print(f"sample {' '.join(sample(W, DICE))}")
    return W


def failures(xs, ys, W):
    # too-high learning rate oscillates instead of descending
    Wh = np.zeros((V, V))
    losses = []
    for _ in range(6):
        Wh = Wh - 50 * grad(Wh, xs, ys)
        losses.append(round(loss_of(Wh, xs, ys), 4))
    assert losses[:3] == [0.4846, 0.7636, 1.1527]
    print(f"lr50    diverges: loss {losses[0]} {losses[1]} {losses[2]} ... (oscillates, no descent)")

    # the zero-frequency problem is gone: an unseen bigram is small, not zero
    P = softmax_rows(W)
    ct = P[IDX["c"], IDX["t"]]
    assert abs(ct - 0.000667) < 1e-5 and ct > 0
    print(f"unseen  c->t P {ct:.6f}  (count model said 0, softmax never does)")


def exit_test(xs, ys):
    W = train(xs, ys, 10, 200, reg=0.10)
    P = softmax_rows(W)
    ac = P[IDX["c"], IDX["a"]]
    ct = P[IDX["c"], IDX["t"]]
    assert abs(ac - 0.340214) < 1e-5 and abs(ct - 0.113314) < 1e-5
    print(f"exit    reg 0.10: a|c {ac:.6f}  c->t {ct:.6f}  (smoothing with a dial)")


def torch_check(xs, ys):
    import torch

    W = torch.zeros((V, V), dtype=torch.float64, requires_grad=True)
    xt, yt = torch.tensor(xs), torch.tensor(ys)
    opt = torch.optim.SGD([W], lr=10.0)
    for _ in range(200):
        opt.zero_grad()
        logits = W[xt]
        logp = logits - torch.logsumexp(logits, 1, keepdim=True)
        loss = -logp[torch.arange(len(xs)), yt].mean()
        loss.backward()
        opt.step()
    P = torch.softmax(W, 1)
    ac = P[IDX["c"], IDX["a"]].item()
    oc = P[IDX["c"], IDX["o"]].item()
    assert abs(loss.item() - 0.277674) < 1e-3
    assert abs(ac - 0.665334) < 1e-3 and abs(oc - 0.331997) < 1e-3
    print(f"torch   loss {loss.item():.6f}  a|c {ac:.6f}  o|c {oc:.6f}  (agrees)")


def main():
    xs, ys = pairs()
    W = headline(xs, ys)
    failures(xs, ys, W)
    exit_test(xs, ys)
    if "--torch" in sys.argv:
        torch_check(xs, ys)


if __name__ == "__main__":
    main()
