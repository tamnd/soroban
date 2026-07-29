# Embeddings, a lookup table you can train

Reference: [word embedding on Wikipedia](https://en.wikipedia.org/wiki/Word_embedding).

An embedding is the plainest idea in a neural network wearing an intimidating name: it is a table of rows, one row per item in some vocabulary, and using it means picking the row for the item you have. Give every letter, word, or token an index, stack a row of numbers for each one into a matrix, and the embedding of item $i$ is just row $i$ of that matrix. The rows are ordinary weights, so gradient descent trains them like any others, which is the whole point: the model learns what each item's row should contain.

## A lookup is a matrix multiply

Here is the fact that ties an embedding to the rest of the math. Let $W$ be a matrix with one row per item, and let $\mathrm{onehot}(i)$ be a row vector that is all zeros except a single $1$ in position $i$. Then

$$\mathrm{onehot}(i) \cdot W = W[i]$$

the whole product is just row $i$ of $W$. The one-hot's single $1$ multiplies row $i$ by one and every other row by zero, so the sum picks out exactly one row. Reading `W[i]` by index and multiplying by a one-hot vector are the same operation; the index form is how you write it in code and the matrix form is how it fits into a network, where the previous layer hands you a one-hot (or a soft, near-one-hot) vector rather than an integer.

## Worked on lesson 0006's matrix

Lesson 0006 uses a $6 \times 6$ matrix over the alphabet `. a b c o t`, so $W$ has thirty-six numbers and each row scores the six possible next letters. The current letter c has index 3, so predicting what follows c means reading row 3:

$$\mathrm{onehot}(3) = (0,0,0,1,0,0), \qquad \mathrm{onehot}(3) \cdot W = W[3] = \text{the c row}.$$

Those six numbers are logits, and a [softmax](softmax.md) turns them into next-letter probabilities. After training, the c row's softmax comes out near $(0, 0.665, 0, 0, 0.332, 0)$: a two-to-one bet on a over o, the same odds lesson 0005 got by counting. The embedding did not learn a meaning for c in the abstract; it learned the row of scores that best predicts what the data puts after c.

## The gradient is sparse

Because a one-hot selects a single row, the gradient touches a single row. Only items that actually appeared in a batch pull on their own rows; every other row gets a gradient of exactly zero and does not move. In lesson 0006 the letters that never appear as a current letter keep their initial weights forever, and the rows that do appear each get a gradient equal to (the model's predicted next-letter frequency) minus (the observed next-letter frequency), the [cross-entropy](cross-entropy.md) gradient $p - y$ summed over that row's examples. Gradient descent is therefore driving each row's softmax toward the frequencies the data actually shows, which is why the trained embedding reproduces the count table. Real models have vocabularies in the tens of thousands and rows hundreds of numbers wide, but the mechanism on this page is the entire story: an embedding is a trainable lookup table, and its gradient only ever edits the rows you looked up.
