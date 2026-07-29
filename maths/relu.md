# ReLU, the switch that lets lines bend

Reference: [rectifier on Wikipedia](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).

Sums of lines are lines. Stack as many neurons as you like, and if each one only multiplies and adds, the whole tower still computes a single straight line, so it can never fit a curve, a corner, or a V. An activation function is the fix: a fixed nonlinear function applied to a neuron's output before the next layer sees it. The one this repo uses, and the one most real networks use, is the rectified linear unit, mercifully shortened to ReLU:

$$\mathrm{relu}(z) = \max(0, z)$$

In words: if the input is positive, pass it through unchanged; if it is negative, output zero. That is the entire definition. On the numbers $z = -2, -0.5, 0.3, 1.5$ it outputs $0, 0, 0.3, 1.5$.

## Read it as a switch

The useful mental model is a switch with two settings. When $z > 0$ the switch is on and the neuron behaves like a plain line. When $z < 0$ the switch is off and the neuron outputs a flat zero, contributing nothing. Which setting a neuron sits in depends on the input, so a network of relu neurons is a collection of lines that turn on and off in different regions, and the sum of such pieces can bend. One kink per neuron is the budget.

## The slope is a gate

Training needs the slope of everything, and relu's slope could not be shorter: it is 1 where the switch is on and 0 where the switch is off. During the backward pass this slope multiplies into the chain, which makes relu act as a gate for gradients. An open gate ($z > 0$) passes the gradient through untouched. A shut gate ($z < 0$) multiplies the gradient by zero, so every parameter behind that gate learns nothing from that data point.

At exactly $z = 0$ the two pieces meet at a corner and the slope is genuinely undefined. Software picks a convention, and this repo, along with torch, uses slope 0 there. Landing on exactly 0.0 in floating point is rare enough that the choice almost never matters, and the lessons choose their numbers so it never happens at all.

## Two relus make a V

One identity shows the power move. For any $x$, either $x$ or $-x$ is negative, so one of the two relus below is always asleep while the other passes its input through:

$$\mathrm{relu}(x) + \mathrm{relu}(-x) = |x|$$

Check it at $x = 2$: the first gives 2, the second gives $\mathrm{relu}(-2) = 0$, sum 2, and $|2| = 2$. At $x = -2$: the first gives 0, the second gives $\mathrm{relu}(2) = 2$, sum 2, and $|-2| = 2$. Each relu owns one arm of the V and stays flat on the other side. Lesson 0002 is this identity turned into a training exercise: a two-neuron hidden layer discovers it by gradient descent.

## Dying ReLU

The gate has a failure mode worth knowing by name. If training pushes a neuron's parameters to where $z < 0$ for every data point, its gate is shut everywhere, so its gradient is zero everywhere, so no update can ever move it back. The neuron is dead, permanently, and the network has lost part of its budget of kinks. This is called the dying ReLU problem, it is usually triggered by a learning rate too large, and lesson 0002's failure gallery manufactures a case on purpose so you can watch both neurons die and the loss flatline while gradient descent runs on, perfectly healthy and perfectly useless.
