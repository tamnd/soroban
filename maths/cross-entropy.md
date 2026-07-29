# Cross-entropy, the price of a probability

Reference: [cross-entropy on Wikipedia](https://en.wikipedia.org/wiki/Cross-entropy).

A classifier outputs a probability for each class, and the truth is one class. Cross-entropy scores that situation with a single number per data point:

$$L = -\ln(p_{\text{true}})$$

the negative [natural log](exp-log.md) of the probability the model gave to the class that was actually correct. The other probabilities enter only through the fact that they took mass away from the true one. Over a dataset, the loss is the mean of these, exactly as [MSE](mean-squared-error.md) was a mean of squared errors.

## Read it as a price list

| $p_{\text{true}}$ | $-\ln(p_{\text{true}})$ | reading |
|---|---|---|
| 1.0 | 0 | certain and right, free |
| 0.99 | 0.010050 | slightly hedged, nearly free |
| 0.9 | 0.105361 | mild hedge, small fee |
| 0.5 | 0.693147 | coin flip |
| 1/3 | 1.098612 | knows nothing (3 classes) |
| 0.01 | 4.605170 | confident and wrong, painful |
| $\to 0$ | $\to \infty$ | certain and wrong, unbounded |

The asymmetry is the design. Between $p = 1$ and $p = 0.9$ the price barely moves; between $p = 0.01$ and $p = 0.001$ it jumps by another $\ln 10 = 2.30$. Being roughly right is cheap and being confidently wrong is ruinous, which is precisely the incentive you want a model trained on this loss to absorb. The $1/3$ row is worth memorizing in general form: a model that knows nothing spreads probability uniformly and pays $\ln(\text{number of classes})$, the starting loss of every classifier that begins from ignorance and the baseline any real learning must beat.

## Why the log

Two reasons, one about meaning and one about mechanics. Meaning: probabilities of independent events multiply, and the log turns that product into a sum, so the loss of a dataset is a sum of per-point losses, each digestible on its own (in that language, minimizing cross-entropy is maximizing the likelihood of the observed labels). Mechanics: the [slope](derivative.md) of $\ln y$ is $1/y$, largest when $y$ is smallest, so the loss pushes hardest exactly on the points the model gets most wrong. MSE has the opposite habit when applied to probabilities that came through a [softmax](softmax.md): its gradient carries a factor of the softmax slope, which flattens toward zero as probabilities pin to 0 or 1, so the worst-classified points are the ones it pushes weakest. Lesson 0003 measures the gap on a concrete point and gets a factor of forty thousand.

## The gradient that falls out

Push the [chain rule](chain-rule.md) through $-\ln(\mathrm{softmax}(z))$ and nearly everything cancels, leaving the friendliest gradient formula in the subject: for each class score $z_k$,

$$\frac{\partial L}{\partial z_k} = p_k - y_k$$

probability minus target, where $y$ is 1 for the true class and 0 for the rest. Overshoot a wrong class by some probability, get pushed down by exactly that much; short the true class, get pushed up by the shortfall. The exp inside softmax and the log inside the loss are inverse functions, and this cancellation is why the pair is used together everywhere; torch fuses them into one function (`cross_entropy`) so the cancellation happens in the code, not only on paper. A bonus of the cancellation: since every $p_k - y_k$ lives between $-1$ and $1$, the gradients cannot blow up no matter how wrong the model is, unlike MSE's, which grow with the error.
