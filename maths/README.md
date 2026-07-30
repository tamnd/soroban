# The maths shelf

Lessons in this repo use mathematical notation because it is the shortest way to state something precisely, but no lesson assumes you can already read it fluently. This folder is the reference shelf: one short page per idea, written for someone meeting that idea for the first time, with the arithmetic worked out on the same numbers the lessons use. When a formula in a lesson stops you, look up its page here, satisfy yourself with a calculator, then go back to the lesson.

| Page | What it covers | First needed in |
|------|----------------|-----------------|
| [notation.md](notation.md) | how to read every symbol that appears: y with a hat, subscripts, the big sigma, the curly d, the arrow, lim | lesson 0001 |
| [mean-squared-error.md](mean-squared-error.md) | how many errors become one number, and why we square | lesson 0001 |
| [derivative.md](derivative.md) | what a slope is, how to measure one by nudging, and what a gradient is | lesson 0001 |
| [chain-rule.md](chain-rule.md) | why the slopes of chained steps multiply | lesson 0001 |
| [gradient-descent.md](gradient-descent.md) | the update rule, the minus sign, and the learning rate | lesson 0001 |
| [floats.md](floats.md) | which decimals a computer stores exactly, and how that decides assert tolerances | lesson 0001 |
| [autodiff.md](autodiff.md) | how torch computes gradients without being given formulas, and why it is called backpropagation | lesson 0001 |
| [computation-graph.md](computation-graph.md) | a formula drawn as wiring, the local-slope table, the backward walk, and why slopes add | lesson 0004 |
| [relu.md](relu.md) | the activation function as a switch, its gradient gate, and the dying ReLU failure | lesson 0002 |
| [exp-log.md](exp-log.md) | the number e, the exponential, the natural log, and why they are inverses | lesson 0003 |
| [softmax.md](softmax.md) | scores into probabilities, shift invariance, and the subtract-max trick | lesson 0003 |
| [cross-entropy.md](cross-entropy.md) | the -ln price list, why the log, and the probability-minus-target gradient | lesson 0003 |
| [bigram.md](bigram.md) | language as counting: the boundary token, counts to probabilities, the loss, sampling, and the zero-frequency problem | lesson 0005 |
| [embedding.md](embedding.md) | a lookup table as a matrix, one-hot times a matrix equals a row, and why the gradient is sparse | lesson 0006 |
| [attention.md](attention.md) | query, key, value, the scaled dot-product score, the causal mask, and the weighted-average output | lesson 0007 |
| [overfitting.md](overfitting.md) | capacity versus data, the train/validation split, the generalization gap, and early stopping | lesson 0008 |

Two habits make these pages work. First, never read past a formula you could not recompute; every formula on this shelf comes with the numbers to recompute it. Second, treat notation as compression rather than difficulty: each symbol is an abbreviation someone invented to avoid writing a sentence over and over, and once you know the sentence, the symbol is your friend.

Pages get added as lessons need them. Each page opens with a link to the long version on Wikipedia if you want more than the working minimum.
