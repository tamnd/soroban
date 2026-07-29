# Floating point, or why 0.1 is not exactly 0.1

Reference: [double-precision floating-point format on Wikipedia](https://en.wikipedia.org/wiki/Double-precision_floating-point_format).

Every number in this repo's training runs is a float64, the 64-bit format nearly all numerical software uses. Floats are binary fractions: a float is some integer times some power of two. That single fact explains every assert tolerance in the lessons, so it is worth thirty seconds.

## Which decimals are exact

A decimal number is exactly representable when it can be written with a power of two in the denominator. Halves, quarters, eighths: $0.5 = 1/2$, $0.25 = 1/4$, $-0.75 = -3/4$, all exact, stored digit for digit. Tenths cannot: $0.1 = 1/10$, and 10 contains a factor of 5 that no power of two can supply. When you type `0.1`, the machine stores the nearest float, which is exactly

```
0.1000000000000000055511151231257827021181583404541015625
```

Python will show you with `from decimal import Decimal; Decimal(0.1)`. The stored value is off by about $5.5 \times 10^{-18}$, invisible at the precision anyone prints, but real, and arithmetic compounds it. The classic demonstration is that `0.1 + 0.2` evaluates to `0.30000000000000004`, so `0.1 + 0.2 == 0.3` is `False` in every language with float64, not a bug in any of them.

## What this means for the lessons

The lessons pin computed numbers against hand arithmetic with asserts, and the float rules decide how each assert is written. When every quantity in a computation is halves and quarters, the machine's arithmetic is exact and the assert uses `==` with no tolerance; lesson 0001's step 1 (loss exactly 41, on integers) and lesson 0002's step 1 (loss exactly 0.25, gradients like $-0.75$ and $-0.5$) are in this regime, and if they mismatch by any amount, someone's arithmetic is wrong. The moment a learning rate like 0.05 or 0.1 enters an update, its tiny representation error rides along into every later step, so asserts from step 2 onward compare with a tolerance of $10^{-9}$: loose enough for accumulated float error, tight enough that a real mistake in the maths (which shows up in the third digit, not the seventeenth) still fires.

The habit to take away: exact equality on floats is not forbidden, it is a claim that the entire computation stays inside the exactly-representable numbers, and you should only write it when you can argue that on paper.
