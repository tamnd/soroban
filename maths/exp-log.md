# The exponential and the natural log

Reference: [exponential function on Wikipedia](https://en.wikipedia.org/wiki/Exponential_function) and [natural logarithm on Wikipedia](https://en.wikipedia.org/wiki/Natural_logarithm).

Two functions run the whole of lesson 0003, and they are inverses of each other: $e^x$ turns any number into a positive one, and $\ln$ turns it back. Both live on every scientific calculator, usually as the `e^x` and `ln` buttons, and this page is what those buttons do.

## The number e and the function e^x

Repeated multiplication is exponentiation: $2^{10} = 1024$ means ten twos multiplied together. The base can be any positive number, and mathematics has a favorite: $e = 2.718281828...$, chosen because the function $e^x$ has the tidiest possible [slope](derivative.md): the slope of $e^x$ is $e^x$ itself, the function's own value, at every point. Nudge-check it at $x = 2$: $e^2 = 7.389056$, and $(e^{2.00000001} - e^2)/0.00000001 = 7.389056$. No other base gets a slope that clean, which is why every formula in this subject that could use any base uses $e$.

Three properties carry everything:

$$e^x > 0 \text{ always} \qquad e^0 = 1 \qquad e^{a+b} = e^a \cdot e^b$$

The first says exponentiating is a machine for making numbers positive: feed it $-2$ and get $e^{-2} = 0.135335$, small but still above zero. The second is the anchor. The third says adding in the exponent is multiplying outside it, and it is the property [softmax](softmax.md) leans on.

Growth is violent in both directions: $e^5 = 148.4$, $e^{10} = 22026.5$, and a [float64](floats.md) gives up past $e^{709} = 8.2 \times 10^{307}$; ask a computer for $e^{710}$ and the answer is `inf`, an overflow that lesson 0003's failure gallery triggers on purpose.

## The natural log

$\ln y$ asks: $e$ to what power gives $y$? So $\ln 1 = 0$, $\ln e = 1$, $\ln 7.389056 = 2$, and $\ln$ of anything at or below zero does not exist, since $e^x$ never gets there. It undoes the exponential exactly: $\ln(e^x) = x$ and $e^{\ln y} = y$.

Its two working properties mirror the exponential's:

$$\ln(a \cdot b) = \ln a + \ln b \qquad \text{slope of } \ln y \text{ is } \frac{1}{y}$$

The first turns multiplication into addition, which is why logs appear wherever probabilities (things that multiply) need to become losses (things that add). The second, nudge-checkable at $y = 4$ where the slope comes out $0.25$, has a consequence worth staring at: as $y$ shrinks toward 0 the slope $1/y$ blows up, so $\ln$ changes fastest exactly where its input is smallest. [Cross-entropy](cross-entropy.md) is built on that behavior.

Useful values to recognize on sight: $\ln 2 = 0.693147$, $\ln 3 = 1.098612$, $\ln 10 = 2.302585$. The middle one is lesson 0003's starting loss, and none of the three is a finite decimal or a finite binary fraction, which is why that lesson's asserts carry tolerances from the very first step.
