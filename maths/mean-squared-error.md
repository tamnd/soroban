# Mean squared error

Reference: [mean squared error on Wikipedia](https://en.wikipedia.org/wiki/Mean_squared_error).

Training needs "how wrong is the model" as one single number, because one number can be compared, plotted, and pushed downhill. The model makes a guess per data point, so we start with one error per point and need a recipe for collapsing many errors into one. Mean squared error is that recipe: square each error, then average.

$$L = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

Read it inside out, using the [notation page](notation.md) if any symbol resists. $\hat{y}_i - y_i$ is the i-th error, guess minus truth. Squaring it gives $(\hat{y}_i - y_i)^2$. The sigma adds the squared errors over all $N$ points, and the $\frac{1}{N}$ divides by the count, making it an average.

Worked on lesson 0001's starting position, where the model guesses 0 for everything and the truths are 3, 5, 7, 9:

```
errors:    0-3 = -3    0-5 = -5    0-7 = -7    0-9 = -9
squares:   9    25    49    81
sum:       164
average:   164 / 4 = 41
```

So $L = 41$. That is the entire computation; if you can produce 41 on a calculator, you understand mean squared error.

## Why square, and not something else

The errors above are all negative, and errors from overshooting would be positive. Adding raw errors lets them cancel: a model that guesses 4 too high on one point and 4 too low on another would score a perfect 0 while being wrong everywhere. Squaring fixes the cancellation, since every square is zero or positive, and a total of zero is achievable only by being exactly right on every point.

Absolute values (dropping the minus signs) would also fix cancellation, and that choice exists under the name [mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error). Squaring wins in this course for two reasons. It punishes big misses disproportionately, since an error of 9 contributes 81 while three errors of 3 contribute 27 in total, so the model attends to its worst mistakes first. And it is smooth everywhere, which matters because training runs on slopes and the absolute value has a kink at zero where the slope is undefined.

## What the number is for

A function that scores wrongness like this is called a [loss function](https://en.wikipedia.org/wiki/Loss_function), and mean squared error is the standard loss when the model predicts numbers. When the model predicts categories a different loss takes over, cross-entropy, which gets its own page when lesson 0003 needs it. Either way the training story is identical: the loss is a single number, smaller is better, and the machinery on the [gradient descent page](gradient-descent.md) exists to shrink it.
