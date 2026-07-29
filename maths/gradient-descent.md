# Gradient descent and the learning rate

Reference: [gradient descent on Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent).

Gradient descent is the procedure that turns slopes into learning, and it fits in one sentence: move every knob a small step against its slope, then repeat. This page unpacks the update rule one piece at a time.

$$w \leftarrow w - \mathrm{lr}\cdot\frac{\partial L}{\partial w}$$

## The arrow

$\leftarrow$ is assignment, like `=` in python: compute the right side with the current $w$, store the result back into $w$. The [notation page](notation.md) covers why an arrow and not an equals sign.

## The minus sign

$\frac{\partial L}{\partial w}$ is the slope of the loss at the current position (measured or computed as on the [derivative page](derivative.md)). A negative slope means the loss falls as $w$ rises, so $w$ should rise; a positive slope means the opposite. Subtracting the slope produces exactly that behavior in both cases, which is worth checking by hand once. In lesson 0001 the slope is -35, so $w \leftarrow 0 - 0.05 \cdot (-35) = 1.75$: the slope was negative, and $w$ went up. The general statement is that the gradient points uphill, so stepping against it steps downhill.

## The learning rate

The slope was measured at one point and is only honest near that point, the way the steepness under your feet says little about the mountain a kilometer away. So we scale the step by a small factor, here $\mathrm{lr} = 0.05$, called the [learning rate](https://en.wikipedia.org/wiki/Learning_rate). It is the single most consequential number you choose when training anything, and lesson 0001 maps its failure modes by experiment:

```
lr = 0.05   losses: 41, 1.13, 0.04, ...          smooth descent
lr = 0.1    losses: 41, 18.41, 8.27, 3.72, ...   overshoots the valley, oscillates, still converges
lr = 0.2    losses: 41, 224, 1229, 6731, ...     each overshoot lands higher, diverges to infinity
```

Too small also fails, in the boring way: the run converges but wastes your patience. The working range sits between the two failures and is found by looking at loss curves, not by theory, which is why the habit of reading those curves matters more than any formula on this shelf.

## The loop

One update per knob, every knob at once, then recompute the loss and its slopes at the new position and go again:

```
repeat:
    compute loss              (mean-squared-error.md)
    compute all slopes        (chain-rule.md)
    knob <- knob - lr * slope (this page)
```

That loop is not an ingredient of training neural networks, it is training. Everything the later lessons add, more knobs, deeper chains, fancier losses, adaptive step sizes, plugs into one of those three lines.
