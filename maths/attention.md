# Attention, a weighted average the data chooses

Reference: [attention on Wikipedia](https://en.wikipedia.org/wiki/Attention_(machine_learning)).

A [bigram](bigram.md) reads one row of a table and stops, so its memory is exactly one token long. Attention is the mechanism that lifts that limit: it lets the prediction at each position be a weighted average of information from every earlier position, with the weights computed from the tokens themselves rather than fixed ahead of time. The model decides, per position and from the data, how much to listen to each token it has already seen.

## Query, key, value

Every position emits three vectors, each a linear function of that position's [embedding](embedding.md) vector $x$:

$$q = x W_q, \qquad k = x W_k, \qquad v = x W_v.$$

Read them as roles in a lookup. The query $q$ is what this position is looking for. The key $k$ is what a position offers to anyone looking back at it. The value $v$ is the payload a position contributes to the output when it is attended to. The matrices $W_q, W_k, W_v$ are ordinary weights, trained by gradient descent like any others, so the model learns what to ask for, what to advertise, and what to pass along.

## The score is a scaled dot product

How much position $i$ attends to an earlier position $j$ is decided by the alignment of $i$'s query with $j$'s key, measured by a dot product and then divided by the square root of the vector length $d$:

$$\text{score}(i, j) = \frac{q_i \cdot k_j}{\sqrt{d}}.$$

The dot product is large and positive when the two vectors point the same way, near zero when they are perpendicular, so it reads as a relevance meter. The division by $\sqrt{d}$ is not decoration. A dot product of two $d$-long vectors is a sum of $d$ products, so its size grows with $d$, and without the scale the scores at large $d$ would push the [softmax](softmax.md) toward a one-hot spike, where its gradient nearly vanishes and learning stalls. Dividing by $\sqrt{d}$ holds the scores at a steady size no matter how wide the vectors get.

## The causal mask: no reading the future

A language model predicts the next token from the ones before it, so position $i$ may attend only to positions $j \le i$. Everything strictly to the right is masked: its score is set to $-\infty$ before the softmax, which sends its weight to exactly zero. Position 0 sees only itself, position 1 sees positions 0 and 1, and so on down the sequence. This is what causal means, and it is why a single forward pass can be trained on every position of a sequence at once without any position cheating by looking ahead.

## The weights and the output

Softmax the allowed scores of position $i$ into weights that are non-negative and sum to one, then take the weighted average of the corresponding values:

$$a_{ij} = \mathrm{softmax}_j\big(\text{score}(i, j)\big), \qquad \text{out}_i = \sum_{j \le i} a_{ij}\, v_j.$$

The output at each position is a blend of the earlier values, mixed in the proportions the scores chose. That is the entire mechanism: score, mask, softmax, weighted sum.

## Worked on three tokens

Take three tokens with two-dimensional embeddings $x_0 = (1, 0)$, $x_1 = (0, 1)$, $x_2 = (1, 1)$, and set $W_q = W_k = W_v = I$ so that $q = k = v = x$ and every remaining number is forced. Here $d = 2$, so the scale is $1/\sqrt{2} = 0.707107$.

Position 2 attends to all three. Its raw dot products are $x_2 \cdot x_0 = 1$, $x_2 \cdot x_1 = 1$, $x_2 \cdot x_2 = 2$, which scale to $0.707107, 0.707107, 1.414214$. Softmax of those is

$$a_2 = (0.248255,\ 0.248255,\ 0.503490),$$

the two equal matches sharing weight evenly and the strongest match, position 2 with itself, taking about half. The output is the weighted sum of the values:

$$\text{out}_2 = 0.248255\,(1,0) + 0.248255\,(0,1) + 0.503490\,(1,1) = (0.751745,\ 0.751745).$$

Position 1 sees only positions 0 and 1, with scores $0$ and $0.707107$, giving weights $(0.330238, 0.669762)$ and output $(0.330238, 0.669762)$. Position 0 sees only itself, weight $1$, output $(1, 0)$. No training has happened; the embeddings and the mask fix every number, and lesson 0007 asserts each one against numpy and torch.

## Why it beats a bigram

A bigram keyed on the current letter cannot tell apart two contexts that share that letter but differ earlier. On the tiny corpus aba, cbc, both words put b in the middle, so a bigram sees b and guesses the next letter at fifty-fifty, paying $\log 2 = 0.693147$ per position it cannot resolve. Attention lets the query at the b position look back to the letter two earlier, a in one word and c in the other, and read off the answer, driving the achievable loss down to $\log(2)/4 = 0.173287$ on that corpus. The gap between those two numbers is the value of the context a bigram throws away and attention keeps. Real transformers stack many such heads and many blocks, but the arithmetic in each head is exactly the score, mask, softmax, weighted sum on this page.
