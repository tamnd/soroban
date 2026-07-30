# FLOPs and the 6ND rule, the cost of a training run

Reference: [neural scaling law on Wikipedia](https://en.wikipedia.org/wiki/Neural_scaling_law).

A FLOP is one floating-point operation, a single multiply or a single add on decimal numbers. Training a neural network is almost entirely multiplies and adds, so counting FLOPs is how you budget a run before it starts: how many operations it needs, and therefore how long a given GPU will take. This page builds the count from one matrix multiply up to the whole run, arriving at the rule that a run over $D$ tokens of a model with $N$ parameters costs about $6ND$ operations.

## One matrix multiply

Multiply an $m \times k$ matrix by a $k \times n$ matrix to get an $m \times n$ result. Each result entry is a dot product of one row against one column, both of length $k$: that is $k$ multiplies and $k-1$ adds, about $2k$ operations. There are $mn$ entries, so the multiply costs

$$\text{FLOPs} = 2 \, m \, k \, n.$$

The $-1$ on the adds is dropped because in a real layer $k$ is in the hundreds and one operation out of a few hundred does not change the budget. Every layer that carries parameters, the projections and the feed-forward, is a matrix multiply of this shape, so this one formula covers the arithmetic that matters.

## Forward pass: about 2N per token

A parameter is a number the model multiplies its input by inside one of those matrix multiplies. Each parameter is used once per token as a multiply followed by an add into a running sum, which is 2 operations. A model with $N$ parameters therefore costs about $2N$ operations to send one token through forward:

$$\text{forward} \approx 2N \text{ per token}.$$

This counts only the parameterized matrix multiplies. The layernorms, the activation, and the softmax are a rounding error next to $2N$ and are dropped, the same way the $-1$ on the adds was.

## Backward pass: about 4N per token

Training needs a gradient for every parameter, and the backward pass computes it. For each weight matrix in the forward pass the backward pass does two matrix multiplies of the same size: one sends the gradient back to the layer's input so the layer below can continue, and one produces the gradient with respect to the weights for the optimizer. Two multiplies the size of the forward one cost twice as much:

$$\text{backward} \approx 4N \text{ per token}.$$

## The whole run: 6ND

Add forward and backward, then multiply by how many tokens the run trains on:

$$\text{cost per token} = 2N + 4N = 6N, \qquad \text{run cost} \approx 6 \, N \, D.$$

Two numbers, the parameter count $N$ and the token count $D$, times 6, give the cost of any training run. For the standard scaling-law form the $N$ used is the non-embedding parameter count, the parameters that actually multiply against every token, which leaves out the position table because it is looked up rather than multiplied through.

## What 6N leaves out

The $6N$ count is a floor because it drops the attention scores, where each token compares itself against every earlier token. That work grows with the context length $T$ rather than with the parameter count, and across a model it adds about $12 \, L \, H \, (D/H) \, T$ operations per token, with $L$ layers and $H$ heads of width $D/H$. For a small model with a long context this term is a real fraction of the total: on GPT-2-124M at context 1024 it is 113246208 operations against the 741920256 of $6N$, about 15 percent on top. The longer the context, the more $6ND$ undercounts, which is why it is a budget floor and not the exact bill.

## From FLOPs to time, and MFU

A GPU advertises a peak rate, operations per second at its fastest. Divide the FLOP budget by that peak to get a floor on wall-clock time. No real run reaches the advertised rate, so the fraction it does reach has a name, model FLOPs utilization:

$$\text{MFU} = \frac{\text{FLOPs the run actually did per second}}{\text{the GPU's advertised peak FLOPs per second}}.$$

An MFU of 1 would mean the card never stalls, which never happens; real training sits somewhere between about a quarter and two-thirds, limited by memory bandwidth, kernel launch overhead, and time spent moving data rather than multiplying. The true wall-clock time is the floor divided by the MFU, so a run at 50 percent MFU takes twice its floor. Lesson 0009 measures the MFU of an RTX 4090 running GPT-2-124M and reconciles the stopwatch with this ledger.
