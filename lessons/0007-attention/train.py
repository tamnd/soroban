"""Lesson 0007: attention, looking further back than one token.

A bigram reads one row of a table and stops, so its memory is one token long. This
lesson introduces attention: one causal self-attention head that lets each position
read every earlier one, score how relevant each is by a scaled query-key dot
product, and mix the values by those weights. The by-hand core is one head over
three fixed tokens, every dot product forced by the embeddings and the causal mask.
The motivation is two loss floors on the corpus aba, cbc: a bigram is stuck at the
coin-flip floor log 2, while a model that sees two letters back reaches log(2)/4.
The experiment trains a tiny one-head transformer on a few thousand characters of
TinyStories and watches its loss fall far below a bigram counted on the same text.

Every asserted number was computed by hand in the spec first. Run it with:

    uv run lessons/0007-attention/train.py
    uv run --with torch lessons/0007-attention/train.py --torch --train

The first seven printed lines match `go run ./cmd/soroban 0007` byte for byte.
"""

import math
import sys

import numpy as np

# The three-token embeddings of the by-hand core: dot products come out whole, so
# every attention weight is a clean softmax you can check with a calculator.
E = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def softmax(scores):
    """The numerically-safe softmax from lesson 0003: subtract the max, then normalize."""
    s = scores - scores.max()
    e = np.exp(s)
    return e / e.sum()


def head(embeddings):
    """One causal self-attention head with identity query/key/value projections.

    Returns, for each position i, the attention weights over positions j <= i and
    the output vector, the weighted sum of the earlier embeddings. The score of
    position j for position i is the dot product E[i].E[j] divided by sqrt(d).
    """
    n, d = embeddings.shape
    scale = 1.0 / math.sqrt(d)
    weights, out = [], []
    for i in range(n):
        scores = np.array([embeddings[i] @ embeddings[j] for j in range(i + 1)]) * scale
        w = softmax(scores)
        o = sum(w[j] * embeddings[j] for j in range(i + 1))
        weights.append(w)
        out.append(o)
    return weights, out


def wrap(word):
    return "." + word + "."


def bigram_floor(corpus):
    """Average cross-entropy of the best possible bigram: count what follows each
    current letter, normalize, and average -log(prob of the true next letter). On
    aba, cbc every current letter is 50/50, so this is exactly log 2."""
    nxt = {}
    for w in corpus:
        s = wrap(w)
        for a, b in zip(s, s[1:]):
            nxt.setdefault(a, {}).setdefault(b, 0)
            nxt[a][b] += 1
    total, n = 0.0, 0
    for w in corpus:
        s = wrap(w)
        for a, b in zip(s, s[1:]):
            row = nxt[a]
            p = row[b] / sum(row.values())
            total += -math.log(p)
            n += 1
    return total / n


def context_floor(corpus):
    """Average cross-entropy of a model that conditions on the whole prefix. On
    aba, cbc only the bare prefix is ambiguous (a starts one word, c the other),
    twice in all, so this is 2*log2 over eight positions, i.e. log(2)/4."""
    cont = {}
    for w in corpus:
        s = wrap(w)
        for k in range(1, len(s)):
            cont.setdefault(s[:k], {}).setdefault(s[k], 0)
            cont[s[:k]][s[k]] += 1
    total, n = 0.0, 0
    for w in corpus:
        s = wrap(w)
        for k in range(1, len(s)):
            row = cont[s[:k]]
            p = row[s[k]] / sum(row.values())
            total += -math.log(p)
            n += 1
    return total / n


def headline():
    """Assert every by-hand number and print the seven-line headline."""
    weights, out = head(E)

    # token 0 attends only to itself; tokens 1 and 2 attend to the past
    assert abs(weights[0][0] - 1.0) < 1e-9
    assert np.allclose(weights[1], [0.330238, 0.669762], atol=1e-6)
    assert np.allclose(weights[2], [0.248255, 0.248255, 0.503490], atol=1e-6)
    assert np.allclose(out[0], [1.0, 0.0], atol=1e-9)
    assert np.allclose(out[1], [0.330238, 0.669762], atol=1e-6)
    assert np.allclose(out[2], [0.751745, 0.751745], atol=1e-6)

    corpus = ["aba", "cbc"]
    bg = bigram_floor(corpus)
    cx = context_floor(corpus)
    assert abs(bg - math.log(2)) < 1e-12
    assert abs(cx - math.log(2) / 4) < 1e-12

    d = E.shape[1]
    scale = 1.0 / math.sqrt(d)
    print(f"head   1 causal self-attention head, 3 tokens, d={d}, scale 1/sqrt(2)={scale:.6f}")
    for i in range(len(E)):
        js = "{" + ",".join(str(j) for j in range(i + 1)) + "}"
        ws = " ".join(f"{x:.6f}" for x in weights[i])
        print(f"tok{i}   attends {js:9s} w {ws:35s} out ({out[i][0]:.6f}, {out[i][1]:.6f})")
    print("task   corpus aba cbc: the letter after b needs the letter TWO back")
    print(f"bigram  floor log2   {bg:.6f}  (after b comes a or c, 50/50, stuck)")
    print(f"context floor log2/4 {cx:.6f}  (two back resolves all but the first letter)")


def torch_check():
    """Build the same three-token head in torch to show the arithmetic is the
    mechanism, not a numpy accident. Token 2's output must match the by-hand one."""
    import torch

    Et = torch.tensor(E)
    n = Et.shape[0]
    scale = 1.0 / math.sqrt(Et.shape[1])
    scores = (Et @ Et.T) * scale
    mask = torch.tril(torch.ones(n, n))
    scores = scores.masked_fill(mask == 0, float("-inf"))
    w = torch.softmax(scores, dim=1)
    o = w @ Et
    assert torch.allclose(o[2], torch.tensor([0.751745, 0.751745], dtype=torch.float64), atol=1e-6)
    print(f"torch   head matches: tok2 out ({o[2, 0].item():.6f}, {o[2, 1].item():.6f})")


# A few thousand characters of TinyStories, lowercased and reduced to letters,
# spaces, and a little punctuation. Small enough to train on a laptop CPU in about
# a minute, large enough that a bigram and an attention model separate cleanly.
CORPUS = """u don't have to be scared of the loud dog, i'll protect you . the mole felt so safe with the little girl. she was very kind and the mole soon came to trust her. he leaned against her and she kept him safe. the mole had found his best friend.
once upon a time, there was a wealthy man named tom. he had a big house near a cliff. tom liked to sort his many toys into different boxes. one sunny day, tom went outside to play with his toys. he took them all out of their boxes and spread them on the ground. he had fun playing with his cars, dolls, and balls. when it was time to go home, tom sorted his toys back into their boxes. he was happy to live in his big house near the cliff. and every day, he played with his toys and sorted them again and again.
once upon a time, there was a cool cat named tom. tom loved to go for a jog in the park. every day, he would put on his cool hat and go for a run. one sunny day, as tom was jogging, he saw a big tree. he decided to turn right and run around it. as he turned, he met a new friend, a dog named sam. sam was also going for a jog in the park. tom and sam jogged together every day. they would turn around the big tree, then sit under it to rest. they became best friends and had lots of fun in the cool park.
once upon a time, there was a dog named spot. spot was a very persistent dog. he loved to play and have fun. one day, spot heard his friends talking. we will celebrate! said one friend. spot was excited. he wanted to celebrate too. he ran to his friends and asked, can i celebrate too? his friends smiled and said, yes, spot! let's all celebrate together! they played games, ate yummy food, and laughed a lot. spot was very happy. his friends were happy too. they all had a great time celebrating. and they all lived happily ever after.
one day, a little girl named lily went to the park. she saw a pretty angel playing. the angel had big wings and a nice smile. lily wanted to catch the angel, but she was too slow. lily called out, angel, please wait for me! but the angel did not hear her. the angel was deaf. she could not hear anything. lily felt sad, but she had an idea. lily picked up a flower and threw it to the angel. the angel saw the flower and smiled at lily. she flew down to lily and they became friends. they played in the park all day, and lily was happy.
once upon a time, there was a long string. this string was very special. it lived in a big, pretty box. the string was very happy in the box. one day, a little boy found the box. he wanted to study the long string. he took the string out of the box and played with it. the string was very happy to be with the little boy. the little boy and the string became best friends. they played all day and had lots of fun. the long string was very happy to have a friend. and they lived happily ever after.
once upon a time, there was a polite bee named bob. bob lived in a big hive with all his bee friends. the hive was in a tall tree, near pretty flowers. bob loved his home. every day, bob and his friends went to the flowers to get food. they took the food back to the hive to store it. they worked together and shared with each other. one day, bob met a new friend, a butterfly named bella. bella was very nice and polite too. they played together and had lots of fun. from that day on, bob and bella were the best of friends.
once upon a time, there was a little beetle named bob. bob was very popular. all his friends liked him a lot. bob lived in a big green tree. one day, bob was playing with his friends when a new beetle appeared. the new beetle was shy and said, hi, i'm tim. can i play too? bob and his friends were happy to have a new friend. they all played together and had so much fun. tim was happy to be with bob and his friends. now, tim was popular too. they all lived happily in the big green tree.
once upon a time, there was a pretty flower. the flower lived in a big garden. one day, a little boy named tim saw the flower. he liked it a lot. tim said, i want to test if the flower can be mine. he picked the flower from the ground. but, oh no! the flower was spoiled. the pretty flower turned brown and sad. tim cried and told his mom, i picked the flower and it got spoiled. his mom said, you should not pick flowers, they are happy in the garden. tim learned that it is better to leave pretty things where they are.
once upon a time, there was a gifted bird named blue. blue could whistle the best songs in the forest. all the other animals loved to hear blue whistle. one day, blue found a shiny black rock. it was coal. blue took the coal to his friend, bunny. bunny liked the coal and used it to draw pictures on the ground. blue and bunny had a fun day playing with the coal and whistling songs. all the animals in the forest came to see the pictures and hear blue whistle. they all had a great time together.
once upon a time, there was an adorable little dog named max. max loved to play with his gear. he had a ball, a bone, and a rope. max played with his gear all day long. one day, max saw a big cat. the cat said, bow to me, little dog. max did not want to bow to the cat, but he did it anyway. the cat laughed and took max's gear. max went home without his gear. he was very sad. his owner tried to make him happy, but max missed his gear too much. the cat never gave max his gear back, and max stayed sad.
once upon a time, in a small house, there was a little girl named mia. mia had a pet bird named bob. bob lived in a cage. mia loved bob very much. one day, mia saw that bob was sad. she asked, bob, why are you sad? bob said, i want to be free and fly. mia felt sad for bob. she wanted to make bob happy. mia opened the cage door and let bob out. bob was very happy. he flew around the room. mia was proud of her bird. she said, bob, i love you! bob flew to mia and let her rub his head. they were both happy and played together all day.
one day, a popular cat named tom went for a walk. he saw a jar on the side of the road. tom was curious, so he stepped closer to look at it. hey, tom, said a bird named sue. what's in the jar? tom didn't know, so he opened the jar. out jumped a tiny frog! they were both surprised. thank you for letting me out, said the frog. i will give you a wish. tom and sue looked at each other. they wished to be friends forever. the frog smiled and their wish came true. they were all very happy.
one day, a boy named tim was very excited. his mom and dad were going to send him to the zoo. he had never been to the zoo before. tim could not wait to see all the animals. when they got to the zoo, tim saw a big lion. the lion roared loud. he also saw a tall giraffe with a long neck. the giraffe ate leaves from the tree. tim was so happy to see all the animals. at the end of the day, tim and his mom and dad went home. tim was very tired but still excited. he told all his friends about the zoo. tim could not wait to go back to the zoo again.
once upon a time, in a peaceful town, there was a big square. in the square, there were many kids who liked to play. they were very happy. one day, a little girl saw a puzzle on the ground. it had many square pieces. she wanted to solve it. so, she asked her friends to help her. they all worked together to solve the puzzle. they put the square pieces in the right place. soon, the puzzle was done. the kids were so happy and proud. they had a fun day in the peaceful town.
one day, a cat and a dog were in a park. the cat was comfortable on a bench. the dog was near a grill. the dog said, i want to play! the cat looked at the dog and said, okay, let's play! they played near the grill. the dog ran fast and the cat jumped high. they had fun. then, the dog's tail hit the grill. ouch! said the dog. the cat ran to help. the cat gave the dog a soft slap on the back. the dog felt better. they went back to play and had a great day.
one day, tim went for a walk with his mom. they saw a big pile of junk near their house. tim was very alert and observed something shiny in the junk. mom, look! tim said. i see something shiny. can i get it? his mom said, okay, but be careful. tim carefully moved the junk and found a toy car. he was so happy. he showed the car to his mom, and she smiled. from then on, tim always observed his surroundings and found many more treasures. he learned that being alert can lead to finding special things.
once upon a time, there was a little boat. the boat liked to go to the shore. one day, the boat saw a big load. the load was heavy and uncomfortable. the boat wanted to help. so, the boat took the load to the shore. the load made the boat very uncomfortable. the boat felt slow and tired. in the end, the boat could not carry the load anymore. the boat stopped moving and stayed on the shore. the boat was sad and uncomfortable forever.
once upon a time, a kind girl named lily went for a walk. she saw a pretty purse on the ground. lily liked the purse and wanted to find who it belonged to. lily asked her friends, do you know who lost this purse? her friends did not know, but they all admired the purse. they thought it was very nice. finally, lily found a sad lady who had lost her purse. the lady was so happy when lily gave it back to her. the lady said, thank you, kind girl! lily smiled and felt good for being helpful."""


def bigram_baseline(data, V):
    """Cross-entropy of a bigram counted on the training text: the baseline to beat."""
    N = np.zeros((V, V))
    for a, b in zip(data, data[1:]):
        N[a, b] += 1
    P = N / np.clip(N.sum(1, keepdims=True), 1, None)
    total = 0.0
    for a, b in zip(data, data[1:]):
        total += -math.log(max(P[a, b], 1e-12))
    return total / (len(data) - 1)


def train_tinystories():
    """Train a one-block, one-head transformer on the TinyStories characters and
    show its loss fall far below a bigram counted on the same text."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(1337)
    text = CORPUS
    chars = sorted(set(text))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    B, T, D = 32, 64, 64
    n = len(data)
    baseline = bigram_baseline(data.tolist(), V)
    assert 2.0 < baseline < 2.3
    print(f"tiny   {text.count(chr(10)) + 1} stories, {len(text)} chars, vocab {V}")
    print(f"bigram baseline on the same text: loss {baseline:.4f}  (the number to beat)")

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = torch.nn.Embedding(V, D)
            self.pos = torch.nn.Embedding(T, D)
            self.q = torch.nn.Linear(D, D, bias=False)
            self.k = torch.nn.Linear(D, D, bias=False)
            self.v = torch.nn.Linear(D, D, bias=False)
            self.proj = torch.nn.Linear(D, D)
            self.ln1 = torch.nn.LayerNorm(D)
            self.ln2 = torch.nn.LayerNorm(D)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(D, 4 * D), torch.nn.GELU(), torch.nn.Linear(4 * D, D)
            )
            self.lnf = torch.nn.LayerNorm(D)
            self.head = torch.nn.Linear(D, V)
            self.register_buffer("mask", torch.tril(torch.ones(T, T)))

        def forward(self, idx):
            Tt = idx.shape[1]
            x = self.tok(idx) + self.pos(torch.arange(Tt))
            h = self.ln1(x)
            att = (self.q(h) @ self.k(h).transpose(-2, -1)) / math.sqrt(D)
            att = att.masked_fill(self.mask[:Tt, :Tt] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            x = x + self.proj(att @ self.v(h))
            x = x + self.mlp(self.ln2(x))
            return self.head(self.lnf(x))

    model = Block()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    nparams = sum(p.numel() for p in model.parameters())

    def batch():
        ix = torch.randint(0, n - T - 1, (B,))
        x = torch.stack([data[i : i + T] for i in ix])
        y = torch.stack([data[i + 1 : i + T + 1] for i in ix])
        return x, y

    loss = None
    for _ in range(3000):
        x, y = batch()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = loss.item()
    assert final < 1.0, final
    print(f"attn   {nparams} params, 3000 steps: train loss {final:.4f}  (well below the bigram)")

    @torch.no_grad()
    def sample(nchars=220, seed="\n"):
        ctx = [stoi[seed]]
        out = []
        for _ in range(nchars):
            idx = torch.tensor([ctx[-T:]])
            p = F.softmax(model(idx)[0, -1], dim=-1)
            nx = torch.multinomial(p, 1).item()
            out.append(itos[nx])
            ctx.append(nx)
        return "".join(out).replace("\n", " ")

    print("sample " + sample())


def main():
    headline()
    if "--torch" in sys.argv:
        torch_check()
    if "--train" in sys.argv:
        train_tinystories()


if __name__ == "__main__":
    main()
