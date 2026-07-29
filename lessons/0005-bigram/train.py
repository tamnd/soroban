#!/usr/bin/env python3
"""Lesson 0005: language as counting, a bigram model over a tiny corpus.

Reads three words, counts which letter follows which, normalizes the counts into
next-letter probabilities, reports the loss, samples new words, and shows the
zero-probability failure. There is no gradient descent here, only tallying and
dividing. The first seven printed lines are the headline table the Go runner
matches byte for byte.

    uv run train.py            # or plain: python3 train.py
    uv run train.py --torch    # cross-check the count matrix and loss in torch
"""

import argparse
import math

import numpy as np

CORPUS = ["cat", "cot", "cab"]
ALPHABET = [".", "a", "b", "c", "o", "t"]
IDX = {c: i for i, c in enumerate(ALPHABET)}
V = len(ALPHABET)
# fixed dice so python, torch, and Go all sample the same words
DICE = [0.5, 0.2, 0.8, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.2, 0.3, 0.5]


def bigrams(word):
    s = "." + word + "."
    return list(zip(s, s[1:]))


def counts(corpus):
    N = np.zeros((V, V), dtype=np.int64)
    for w in corpus:
        for a, b in bigrams(w):
            N[IDX[a], IDX[b]] += 1
    return N


def probs(N):
    rows = N.sum(axis=1, keepdims=True)
    return np.divide(N, rows, out=np.zeros((V, V)), where=rows > 0)


def word_prob(P, word):
    p = 1.0
    for a, b in bigrams(word):
        if a not in IDX or b not in IDX:
            return 0.0  # a symbol the corpus never saw: unseen bigram
        p *= P[IDX[a], IDX[b]]
    return p


def nll(P, corpus):
    total, n = 0.0, 0
    for w in corpus:
        for a, b in bigrams(w):
            total += -math.log(P[IDX[a], IDX[b]])
            n += 1
    return total / n


def sample(P, dice, n_words=3):
    it = iter(dice)
    words = []
    for _ in range(n_words):
        cur, out = ".", []
        while True:
            u = next(it)
            cum = 0.0
            nxt = "."
            for s in ALPHABET:
                cum += P[IDX[cur], IDX[s]]
                if cum >= u:
                    nxt = s
                    break
            if nxt == ".":
                break
            out.append(nxt)
            cur = nxt
        words.append("".join(out))
    return words


def headline():
    """The seven lines the Go runner reproduces byte for byte."""
    N = counts(CORPUS)
    P = probs(N)

    print("corpus " + " ".join(CORPUS))
    print("alphabet " + " ".join(ALPHABET))

    cs, ps = [], []
    for a in ALPHABET:
        for b in ALPHABET:
            if N[IDX[a], IDX[b]] > 0:
                cs.append(f"{a}{b} {N[IDX[a], IDX[b]]}")
                ps.append(f"{a}{b} {P[IDX[a], IDX[b]]:.3f}")
    print("counts " + "  ".join(cs))
    print("probs " + "  ".join(ps))
    print(f"loss {nll(P, CORPUS):.6f}  (log3/4, every word 1/3)")
    print("sample " + " ".join(sample(P, DICE)))
    print(f"holdout dog P {word_prob(P, 'dog'):.6f}  loss infinite  (unseen .d)")

    # asserts: every headline number matches the hand table in 01-by-hand.md
    assert N.sum() == 12
    assert (N[IDX["."], IDX["c"]], N[IDX["c"], IDX["a"]], N[IDX["c"], IDX["o"]]) == (3, 2, 1)
    assert P[IDX["c"], IDX["a"]] == 2 / 3 and P[IDX["c"], IDX["o"]] == 1 / 3
    assert P[IDX["a"], IDX["t"]] == 0.5 and P[IDX["a"], IDX["b"]] == 0.5
    assert abs(nll(P, CORPUS) - math.log(3) / 4) < 1e-12
    for w in CORPUS:
        assert abs(word_prob(P, w) - 1 / 3) < 1e-12
    assert sample(P, DICE) == ["cat", "cot", "cab"]
    assert word_prob(P, "dog") == 0.0
    return P


def smoothing_check(N):
    """Add-one smoothing rescues the unseen word from an infinite loss."""

    def psmooth(a, b):
        c = N[IDX[a], IDX[b]] if (a in IDX and b in IDX) else 0
        row = N[IDX[a]].sum() if a in IDX else 0
        return (c + 1) / (row + V)

    p = 1.0
    for a, b in bigrams("dog"):
        p *= psmooth(a, b)
    print(f"smoothing dog P {p:.6f}  loss {-math.log(p):.6f}  (add-one)")
    assert abs(p - 0.00044092) < 1e-8


def exit_test():
    """Corpus with cab twice: the model gets more certain and the loss drops."""
    N = counts(["cat", "cot", "cab", "cab"])
    P = probs(N)
    loss = nll(P, ["cat", "cot", "cab", "cab"])
    print(f"exit    cab twice: a|c {P[IDX['c'], IDX['a']]:.3f}  "
          f"o|c {P[IDX['c'], IDX['o']]:.3f}  loss {loss:.6f}")
    assert P[IDX["c"], IDX["a"]] == 0.75 and P[IDX["c"], IDX["o"]] == 0.25
    assert abs(loss - 0.259930) < 1e-6
    assert loss < math.log(3) / 4  # more certain, lower loss


def torch_check():
    import torch

    N = torch.zeros((V, V), dtype=torch.float64)
    for w in CORPUS:
        for a, b in bigrams(w):
            N[IDX[a], IDX[b]] += 1
    P = N / N.sum(1, keepdim=True).clamp(min=1)
    prev = torch.tensor([IDX[a] for w in CORPUS for a, _ in bigrams(w)])
    nxt = torch.tensor([IDX[b] for w in CORPUS for _, b in bigrams(w)])
    loss = (-P[prev, nxt].log()).mean().item()
    print(f"torch   loss {loss:.6f}  (agrees)")
    assert abs(loss - math.log(3) / 4) < 1e-12


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--torch", action="store_true", help="cross-check in torch")
    args = ap.parse_args()

    P = headline()
    smoothing_check(counts(CORPUS))
    exit_test()
    if args.torch:
        torch_check()
