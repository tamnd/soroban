# Rank and the low-rank update, how LoRA fine-tunes cheaply

Reference: [low-rank adaptation on Wikipedia](https://en.wikipedia.org/wiki/Fine-tuning_(deep_learning)#Low-rank_adaptation).

Fine-tuning takes a model someone else trained and adjusts its weights for a new task. The naive way learns a full-size change to every weight matrix, which for a seven-billion-parameter model needs about a hundred gigabytes of memory and so cannot run on one consumer card. LoRA makes the change small by forcing it through a low rank, and QLoRA shrinks the frozen part on top. This page builds the idea from what rank means up to the exact parameter and memory counts lesson 0010 works by hand.

## What rank means

A matrix maps input vectors to output vectors. Its rank is the number of independent directions its output can span: a 4096-by-4096 matrix can have rank up to 4096, but it need not. If a matrix has rank $r$, every output it produces lives in an $r$-dimensional subspace, no matter that the matrix is written as a full 4096-by-4096 grid. A low-rank matrix carries less information than its shape suggests, and that is the opening LoRA uses: a fine-tuning update, it turns out, does not need full rank to be useful.

## Factoring a low-rank matrix

Any matrix of rank $r$ can be written as a product of two thin matrices. An update $\Delta W$ of shape $d \times k$ and rank $r$ factors as

$$\Delta W = B A, \qquad B \text{ is } d \times r, \quad A \text{ is } r \times k.$$

The product of a $d \times r$ and an $r \times k$ matrix is $d \times k$, the right shape, but $B$ and $A$ together hold only $r(d + k)$ numbers instead of $d \times k$. When $r$ is small this is a large saving: for a 4096-by-4096 weight at $r = 16$ it is $16 \times (4096 + 4096) = 131072$ numbers against $4096 \times 4096 = 16777216$, about 0.78 percent. LoRA freezes the original weight $W$ and trains only $B$ and $A$, so the effective weight is $W + BA$ with just $r(d+k)$ trainable numbers per matrix.

## Why it starts as a no-op

Attaching an untrained adapter must not disturb the pretrained model, or the weights are corrupted before any learning happens. LoRA guarantees this by initializing $B$ to all zeros and $A$ to small random values. Then

$$BA = 0 \cdot A = 0,$$

so at step zero the effective weight is $W + 0 = W$, the base unchanged. The first gradient flows into $A$ and $B$, $B$ leaves zero, and the adapter begins to bend the output. This is why LoRA can be bolted onto any pretrained model with no warm-up: on the forward pass at step zero it is the identity.

## Counting the adapter

The trainable count is the per-matrix count $r(d+k)$ summed over the matrices the adapter is attached to. For Llama-2-7B, hidden width $H = 4096$ and MLP width $I = 11008$, LoRA is usually put on the seven linear layers in each of the 32 blocks: the four attention projections q, k, v, o of shape $H \times H$, and the three feed-forward projections gate and up of shape $H \times I$ and down of shape $I \times H$. Per block that is

$$4 \cdot r(H + H) + 2 \cdot r(H + I) + r(I + H) = 4 \cdot 131072 + 2 \cdot 241664 + 241664 = 1249280,$$

and across 32 blocks $32 \times 1249280 = 39976960$ trainable parameters, about 0.59 percent of the 6738415616-parameter base. The other 99.41 percent stays frozen. Rank is the one knob: doubling $r$ doubles the adapter, halving it halves the adapter, and the fraction of the model you train follows directly.

## The memory ledger, and QLoRA

Training memory is a parameter count times bytes per parameter, the same accounting as the FLOP budget in [flops.md](flops.md). A full fine-tune in mixed precision keeps, per parameter, a 2-byte weight, a 2-byte gradient, and the Adam optimizer's two 4-byte moments plus a 4-byte master copy, about 16 bytes. For 6.74 billion parameters that is

$$6738415616 \times 16 \approx 107.8 \text{ GB},$$

more than four times a 24 GB card, before a single activation. This does not shrink with batch size: it is fixed by the parameter count, so you cannot batch your way onto one card, you have to train fewer numbers. LoRA already does that for the trainable part. QLoRA handles the frozen part: since the base weights only get read, never updated, they are quantized to four bits, half a byte each, so the base costs $6738415616 \times 0.5 \approx 3.37$ GB. The adapters carry optimizer state, but there are only 39976960 of them, about 0.64 GB at 16 bytes each. Base plus adapters is about 4 GB, and even with activations the run fits one 24 GB 4090, which is the whole point: a hundred-gigabyte job made to run on one card by training a low-rank fraction of a four-bit-frozen model.
