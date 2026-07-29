# The chain rule

Reference: [chain rule on Wikipedia](https://en.wikipedia.org/wiki/Chain_rule).

The chain rule answers one question: when a change flows through several steps in a row, what is the overall slope? The answer is that the slopes of the steps multiply, and everything called [backpropagation](https://en.wikipedia.org/wiki/Backpropagation) is this fact applied at industrial scale.

## Gears first

Two gears are linked so that turning the first by 1 tooth turns the second by 3, and the second is linked to a third so that 1 tooth of the second moves the third by 2. Turn the first gear by 1 tooth and the third moves by 6, because the effects compose: 1 becomes 3 becomes 6. Slopes chain the same way, since a slope is a gear ratio between a knob and an output: 3 times 2 is 6, and that multiplication is the entire chain rule.

## The same thing with lesson 0001's numbers

At the start of lesson 0001, take the single data point x = 2, y = 5. The point's contribution to the loss is built in two steps:

```
step 1:  e = w * x + b - y       (the error; at w = 0, b = 0 it is -5)
step 2:  contribution = e^2      (the squared error, 25)
```

A nudge to $w$ has to travel through both steps to reach the loss, so find each step's gear ratio. Step 1: raising $w$ by a hair $h$ raises $e$ by $h \cdot x$, so the slope of $e$ with respect to $w$ is $x$, which is 2 here. Step 2: the slope of $e^2$ with respect to $e$ is $2e$ (nudge $e$ by $h$ and $(e+h)^2 = e^2 + 2eh + h^2$, so the change is $2eh$ plus an $h^2$ term too small to matter), which at $e = -5$ is -10. Multiply the ratios:

```
slope of e^2 with respect to w  =  (slope of e^2 wrt e) * (slope of e wrt w)
                                =  2e * x  =  (-10) * 2  =  -20
```

Not convinced by the algebra? Nudge and check, as the [derivative page](derivative.md) taught: at $w = 0.001$ this point's squared error is $(0.002 - 5)^2 = 24.980004$, the change from 25 is -0.019996, and dividing by 0.001 gives -19.996, which is -20 up to the nudge's bluntness.

## From one point to the gradient

The loss averages the contributions of all $N$ points, and the slope of an average is the average of the slopes. Each point contributes slope $2 e_i x_i$ for $w$ and $2 e_i$ for $b$ (the $b$ chain is identical except that step 1's ratio is 1, since raising $b$ by $h$ raises $e$ by exactly $h$). Averaging gives the gradient formulas lesson 0001 uses:

$$\frac{\partial L}{\partial w} = \frac{2}{N}\sum_{i=1}^{N} e_i x_i \qquad \frac{\partial L}{\partial b} = \frac{2}{N}\sum_{i=1}^{N} e_i$$

Nothing in those formulas was guessed; each factor is one gear ratio from the two-step chain, and the $\frac{2}{N}\sum$ wrapper is the averaging (readable symbol by symbol via the [notation page](notation.md)).

## Why this one rule scales to GPT

A deep network is a longer chain, dozens or hundreds of steps between a knob and the loss instead of two, and the rule never changes: multiply the ratios along the path. [Automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) is software that records the steps as they run and does the multiplying for you, walking the chain backward from the loss, and lesson 0004 builds one from scratch. When you meet it, remember it is doing nothing you have not already done here with a calculator.
