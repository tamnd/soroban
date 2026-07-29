# Automatic differentiation and backpropagation

Reference: [automatic differentiation on Wikipedia](https://en.wikipedia.org/wiki/Automatic_differentiation).

Lesson 0001 computes $\frac{\partial L}{\partial w}$ by deriving a formula on paper with the [chain rule](chain-rule.md). That works for two knobs and fails as an approach for millions, not because the maths changes but because nobody can write out a million formulas without error. Automatic differentiation is the observation that nobody has to: the deriving itself is mechanical, so the machine can do it.

## Recording, then replaying backwards

The trick has two halves. First, while the loss is being computed forwards, record every elementary operation (this times that, this plus that, this squared) into a graph whose nodes are intermediate values and whose edges say which values fed which operation. Every one of those elementary operations has a slope you already know from a table: the slope of $e^2$ with respect to $e$ is $2e$, the slope of $w \cdot x$ with respect to $w$ is $x$.

Second, walk the recording in reverse, from the loss back toward the inputs, applying the chain rule at each node: the slope of the loss with respect to any value is the slope one link downstream, times the local slope of the operation between them. By the time the walk reaches a parameter, the product of local slopes along the way is exactly the derivative you would have derived on paper. This backward walk is why the procedure is called backpropagation: the loss's sensitivity propagates backwards through the graph, one multiplication per link.

Nobody types a gradient formula anywhere. You write only the forward computation, and the machine returns $-35$ and $-12$ for lesson 0001, the same numbers the hand derivation gives, because it performed the same derivation.

## The same numbers, three ways

The lessons exploit this for cross-checking. The same gradient can be produced by hand formula, by a nudge measurement (the [finite difference](derivative.md) test), and by autodiff, and the three must agree: formula and autodiff to the last bit, since they do identical arithmetic, and the nudge to the accuracy of the nudge. When gradient code breaks, the nudge test catches it, which is why this repo's Go engine ships with one.

## The engine in this repo

Torch's autograd is an industrial version of this recording-and-replaying, and the `grad` package in this repo is a minimal one: a `Value` holds a number, remembers which values produced it and how, and `Backward()` walks the graph in reverse accumulating slopes into each `Grad` field. Lesson 0004 builds it up from nothing, at which point this page stops being a description and becomes a specification.
