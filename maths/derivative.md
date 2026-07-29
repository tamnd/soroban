# Derivatives, by nudging

Reference: [derivative on Wikipedia](https://en.wikipedia.org/wiki/Derivative).

A derivative is a slope, and a slope is something you can measure with a calculator: change the input a little, watch how much the output changes, divide. No course prerequisite hides in that sentence, and this page builds the whole idea from one measurement.

## The nudge experiment

Take lesson 0001 at its starting point. The loss is a machine: feed in a value of $w$, out comes a number. At $w = 0$ it outputs 41. Nudge $w$ up by 0.001 and recompute: the loss comes out 40.9650075. Now divide the output change by the input change:

```
slope = (40.9650075 - 41) / 0.001 = -34.9925
```

That ratio is the slope of the loss at $w = 0$, and it answers the only question training ever asks: which way, and how strongly, does the output respond to this knob. Negative slope means raising $w$ lowers the loss. A slope near -35 means the response is strong, roughly 35 units of loss per unit of $w$, for small moves.

## What the notation adds

The formal definition dresses the same experiment in symbols:

$$\frac{\partial L}{\partial w} = \lim_{h \to 0} \frac{L(w+h) - L(w)}{h}$$

Match the pieces against the experiment: $h$ is the nudge (we used 0.001), $L(w+h) - L(w)$ is the output change (we got -0.0349925), the fraction is the divide we did, and $\lim_{h \to 0}$ says to imagine the nudge shrinking toward zero (the [notation page](notation.md) unpacks each symbol individually). Our measured -34.9925 is slightly off from the exact answer of -35 only because 0.001 is not infinitely small. The derivative is the destination those measurements approach.

The curly $\partial$ instead of a straight d flags that $L$ has more than one knob ($w$ and $b$ here) and we nudged only one, holding the rest still. Such a slope is called a [partial derivative](https://en.wikipedia.org/wiki/Partial_derivative), partial because it tells only one knob's share of the story.

## The gradient is all the slopes at once

Run the same nudge on $b$ and you measure -11.999, so the exact slope is -12. Collect one slope per knob into a list and you have the [gradient](https://en.wikipedia.org/wiki/Gradient): here, (-35, -12). The gradient is the model's full steering information, one "which way is downhill and how steeply" reading per knob, and for a model with millions of knobs it is a list of millions of slopes, each meaning exactly what the two here mean.

## Why formulas replace nudging

Nudging is honest but costs one full loss computation per knob per step, which is ruinous at millions of knobs. The [chain rule](chain-rule.md) computes the identical numbers directly from the structure of the formula, and lesson 0001 checks the two methods against each other: formula says -35, nudge says -34.9925, agreement to the accuracy of the nudge. That checking habit is worth keeping, since a nudge test catches broken gradient code at every scale.

One caution before trusting a slope too far. The measurement is local, made at one point, and says nothing about the landscape far away. That is why the [update rule](gradient-descent.md) that uses these slopes moves in small steps.
