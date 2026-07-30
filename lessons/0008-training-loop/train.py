"""Lesson 0008: the training loop as instrument.

Every earlier lesson watched a single number, the training loss, fall and called
that success. This lesson shows why that number, on its own, lies. A model with
more parameters than it has data points can drive its training loss to zero by
memorizing, and the training loss cannot tell memorizing from learning. The by-hand
core fits three noisy points with a straight line (two parameters) and with a
parabola (three parameters, one per point): the parabola's training error is exactly
zero, yet on two held-out points it is 328 times worse than the line's. The gap
between training error and held-out error is overfitting, and the held-out set is
the instrument that measures it.

The experiment turns the same instrument on lesson 0007's tiny transformer. Split
the TinyStories text into a training part and a held-out part, train three models of
increasing width, and watch the same story at scale: wider models push the training
loss lower and the held-out loss higher, and the held-out loss traces a U, bottoming
early and then climbing as the model starts memorizing. Where it bottoms is where you
should have stopped.

Every asserted number was computed by hand in the spec first. Run it with:

    uv run lessons/0008-training-loop/train.py
    uv run --with torch lessons/0008-training-loop/train.py --train

The first six printed lines match `go run ./cmd/soroban 0008` byte for byte.
"""

import math
import sys

import numpy as np

# The by-hand dataset. The true relationship is the line y = 0.5x + 0.5. It is
# sampled at x = 0, 1, 2, but the middle sample is nudged off the line to 1.3 (its
# true value is 1.0): that nudge is the noise. Two clean held-out points sit on the
# true line at x = 3, 4. A line fit ignores the nudge; a parabola bends to hit it.
TRAIN_X = np.array([0.0, 1.0, 2.0])
TRAIN_Y = np.array([0.5, 1.3, 1.5])
VAL_X = np.array([3.0, 4.0])
VAL_Y = np.array([2.0, 2.5])


def mse(coef, x, y):
    """Mean squared error of a polynomial fit (numpy coefficient order) on points."""
    return float(np.mean((np.polyval(coef, x) - y) ** 2))


def param_count(V, D, T):
    """Trained parameters in lesson 0007's one-block, one-head transformer: token and
    position embeddings, the four attention projections, two feed-forward layers, the
    three layer norms, and the output head."""
    tok = V * D
    pos = T * D
    qkv = 3 * (D * D)
    proj = D * D + D
    mlp = (D * 4 * D + 4 * D) + (4 * D * D + D)
    norms = 3 * (2 * D)
    head = D * V + V
    return tok + pos + qkv + proj + mlp + norms + head


def headline():
    """Assert every by-hand number and print the six-line headline."""
    line = np.polyfit(TRAIN_X, TRAIN_Y, 1)
    parab = np.polyfit(TRAIN_X, TRAIN_Y, 2)

    # least-squares line y = 0.5x + 0.6; interpolating parabola y = -0.3x^2 + 1.1x + 0.5
    assert np.allclose(line, [0.5, 0.6], atol=1e-9), line
    assert np.allclose(parab, [-0.3, 1.1, 0.5], atol=1e-9), parab

    line_tr, line_va = mse(line, TRAIN_X, TRAIN_Y), mse(line, VAL_X, VAL_Y)
    parab_tr, parab_va = mse(parab, TRAIN_X, TRAIN_Y), mse(parab, VAL_X, VAL_Y)
    assert abs(line_tr - 0.02) < 1e-12 and abs(line_va - 0.01) < 1e-12
    assert abs(parab_tr) < 1e-9 and abs(parab_va - 3.285) < 1e-9

    params = param_count(33, 64, 64)
    assert params == 58273, params
    assert abs(params / 9311.0 - 6.258511) < 1e-6

    print("fit    3 train points off a line, 2 held out; true line y=0.5x+0.5")
    print(f"line   2 params: train mse {line_tr:.6f}, val mse {line_va:.6f}  (generalizes)")
    print(f"parab  3 params: train mse {parab_tr:.6f}, val mse {parab_va:.6f}  (memorized the noise)")
    print("read   lowest train mse picks parab; the val set exposes it as the worse model")
    print(f"count  transformer {params} params on 9311 chars = {params / 9311.0:.3f} params per char")
    print("regime params exceed data points, so memorizing is possible: watch held-out loss")


# The TinyStories characters from lesson 0007, embedded so this lesson is
# self-contained: lowercased, reduced to letters, spaces, and a little punctuation.
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

# The width sweep and the training schedule. B, T are the batch and context sizes,
# widths are the model dimension D of each of the three models, and BASELINE is the
# bigram cross-entropy on this text from lesson 0007, the line the held-out loss is
# measured against.
B, T = 32, 64
WIDTHS = [16, 32, 64]
STEPS, EVAL = 3000, 150
BASELINE = 2.1651


def build_model(V, D, torch):
    """Lesson 0007's transformer at width D: token and position embeddings, one
    causal head, a feed-forward block, layer norms, and a linear head to logits."""
    import torch.nn.functional as F

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

    return Block()


def train_val():
    """Split the text, train each width, and watch the train/val gap open."""
    import torch
    import torch.nn.functional as F

    chars = sorted(set(CORPUS))
    V = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    full = torch.tensor([stoi[c] for c in CORPUS], dtype=torch.long)
    n = len(full)
    cut = int(0.9 * n)
    train_data, val_data = full[:cut], full[cut:]
    print(f"split  {len(train_data)} train chars, {len(val_data)} val chars, vocab {V}")
    print("sweep  wider models push train loss down and held-out loss up:")

    def get_batch(data, g):
        ix = torch.randint(0, len(data) - T - 1, (B,), generator=g)
        x = torch.stack([data[i : i + T] for i in ix])
        y = torch.stack([data[i + 1 : i + T + 1] for i in ix])
        return x, y

    def run(D, keep_curve):
        torch.manual_seed(1337)
        model = build_model(V, D, torch)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        g = torch.Generator().manual_seed(1337)
        nparams = sum(p.numel() for p in model.parameters())

        @torch.no_grad()
        def est(data):
            ge = torch.Generator().manual_seed(0)
            losses = []
            for _ in range(50):
                x, y = get_batch(data, ge)
                losses.append(F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1)).item())
            return sum(losses) / len(losses)

        best_v, best_s, curve, final_t, final_v = 1e9, 0, [], 0.0, 0.0
        for step in range(STEPS + 1):
            if step % EVAL == 0:
                tl, vl = est(train_data), est(val_data)
                if keep_curve:
                    curve.append((step, round(tl, 4), round(vl, 4)))
                if vl < best_v:
                    best_v, best_s = vl, step
                if step == STEPS:
                    final_t, final_v = tl, vl
            if step < STEPS:
                x, y = get_batch(train_data, g)
                loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
        return dict(D=D, params=nparams, final_t=final_t, final_v=final_v,
                    best_v=best_v, best_s=best_s, curve=curve)

    rows = [run(D, keep_curve=(D == WIDTHS[-1])) for D in WIDTHS]
    for r in rows:
        print(f"  D={r['D']:2d}  {r['params']:5d} params  {r['params'] / len(train_data):.2f}/char"
              f"  final train {r['final_t']:.3f}  final val {r['final_v']:.3f}"
              f"  best val {r['best_v']:.3f} @ step {r['best_s']}")
    big = rows[-1]
    print("read   more params: final train falls, final val rises; the widening gap is overfitting")
    print(f"early  the D=64 val bottoms at step {big['best_s']} then climbs to {big['final_v']:.3f}; stop at the minimum")
    print(f"vs bg  early-stopped val {big['best_v']:.3f} beats bigram {BASELINE}; trained to the end {big['final_v']:.3f} loses to it")

    # The gap is real and orders correctly: more capacity, lower train, higher val.
    assert rows[0]["final_t"] > rows[1]["final_t"] > rows[2]["final_t"], "train loss should fall with width"
    assert rows[0]["final_v"] < rows[1]["final_v"] < rows[2]["final_v"], "final val loss should rise with width"
    assert big["final_v"] - big["final_t"] > 3.0, big
    assert big["best_s"] <= 900, big["best_s"]
    assert big["best_v"] < BASELINE < big["final_v"], big
    return rows


def main():
    headline()
    if "--train" in sys.argv:
        train_val()


if __name__ == "__main__":
    main()
