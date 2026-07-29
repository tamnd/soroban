# Softmax, scores into probabilities

Reference: [softmax function on Wikipedia](https://en.wikipedia.org/wiki/Softmax_function).

A classifier's raw outputs are scores, one per class, called logits: any real numbers, positive or negative, in no particular range. Probabilities have rules: each between 0 and 1, all summing to 1. Softmax is the standard bridge:

$$\mathrm{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

In words: [exponentiate](exp-log.md) every score, which makes them all positive while keeping their order, then divide each by the total, which makes them sum to 1. Bigger score, bigger probability, always.

## A worked example

Scores $z = (2, 0, -2)$. Exponentials: $e^2 = 7.389056$, $e^0 = 1$, $e^{-2} = 0.135335$. Their sum is $8.524391$. Divide through:

$$p = (0.866813, \ 0.117310, \ 0.015876)$$

Check they sum to 1. Notice what softmax did with the gaps: a lead of 2 points in score became a lead of $e^2 = 7.39$ times in probability, because $e^{a}/e^{b} = e^{a-b}$. Softmax is run on score differences; that is the next fact, made exact.

## Shift invariance

Add any constant $c$ to every score and nothing happens: the numerator and denominator both pick up a factor $e^c$, which cancels.

$$\mathrm{softmax}(z + c) = \mathrm{softmax}(z)$$

So $(2, 0, -2)$, $(102, 100, 98)$, and $(1000, 998, 996)$ all describe the same three probabilities. Only the gaps between scores matter, and there is always a spare degree of freedom in the logits that the probabilities ignore.

## The subtract-max trick

Shift invariance is also a survival rule. Ask a computer for softmax of $(1000, 998, 996)$ the literal way and $e^{1000}$ [overflows](exp-log.md) to `inf`, the division computes `inf/inf`, and every probability comes back `nan`, poisoning everything downstream. The fix costs one line: subtract the maximum score from all of them first, which shift invariance says changes nothing. Now the scores are $(0, -2, -4)$, the largest exponential is $e^0 = 1$, overflow is impossible, and the answer is the correct $(0.866813, 0.117310, 0.015876)$. Every serious implementation does this, torch's included, and lesson 0003's failure gallery shows a training run where the naive version dies of `nan` mid-run while the subtract-max version sails through the same numbers.

## Why "soft" max

Hardmax would put probability 1 on the biggest score and 0 elsewhere: a verdict with no slope, useless for [gradient descent](gradient-descent.md). Softmax gives the biggest score most of the probability but leaves the rest a share that shrinks smoothly as the gaps grow, and smooth means trainable. Stretch the gaps ($z = (20, 0, -20)$) and softmax approaches the hard verdict; shrink them ($z = (0.2, 0, -0.2)$) and it hovers near uniform. What softmax reports is not just a winner but how sure the scores are about it, and [cross-entropy](cross-entropy.md) is how that confidence gets priced.
