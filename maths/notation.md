# How to read the notation

Reference: [Wikipedia's glossary of mathematical symbols](https://en.wikipedia.org/wiki/Glossary_of_mathematical_symbols).

Mathematical notation is a set of abbreviations. Every symbol below stands for a short sentence, and once you can say the sentence, the formula reads like ordinary text. All examples use lesson 0001's data: inputs x = 1, 2, 3, 4 and answers y = 3, 5, 7, 9.

## Letters are containers with values inside

A letter like $w$ or $b$ is a name for a number whose value we do not want to fix yet, the same way a variable works in code. When a lesson says "at $w = 1.75$", it means: everywhere you see $w$, put 1.75.

## The hat: $\hat{y}$

A hat on a letter marks a guess or estimate of that letter. So $y$ is the true answer from the data and $\hat{y}$ (say "y hat") is the model's guess at it. The distance between them is the whole subject of training.

## Subscripts: $x_i$

A subscript picks one item out of a list. Our inputs are a list of four numbers, and $x_i$ means "the i-th input". So $x_1 = 1$, $x_2 = 2$, $x_3 = 3$, $x_4 = 4$. The letter $N$ conventionally holds the length of the list, so here $N = 4$. In code, $x_i$ is `x[i]` and $N$ is `len(x)`.

## The big sigma: $\sum$

The symbol $\sum$ (a capital Greek sigma, for Sum) means "add these up". The decorations tell you what to add and over what range:

$$\sum_{i=1}^{N} x_i$$

reads "let $i$ run from 1 to $N$, and add up the $x_i$ you meet". On our data that is $1 + 2 + 3 + 4 = 10$. It is a for loop with an accumulator, nothing more:

```python
total = 0
for i in range(1, N + 1):
    total += x[i]
```

Whatever expression sits after the sigma is what gets accumulated. For example $\sum_{i=1}^{N} x_i^2 = 1 + 4 + 9 + 16 = 30$.

## Averages: $\frac{1}{N}\sum$

An average is a sum divided by the count, so the recipe "average of the squared errors" comes out as $\frac{1}{N}\sum_{i=1}^{N} e_i^2$. Reading tip: when you see $\frac{1}{N}\sum$, say "the average of" and move on.

## The curly d: $\frac{\partial L}{\partial w}$

This one causes the most fear and abbreviates the friendliest idea. $\frac{\partial L}{\partial w}$ (say "dee L dee w") is a single symbol, not a fraction to compute, and it means: the slope of $L$ with respect to $w$, in other words how fast $L$ changes when you nudge $w$ a tiny bit while holding everything else still. The curly $\partial$ rather than a straight d signals that $L$ has several knobs and we are nudging only one of them. The full story, including how to measure this number with a calculator, is on the [derivative page](derivative.md).

## The limit: $\lim_{h \to 0}$

$\lim_{h \to 0}$ reads "as h shrinks toward zero". It appears in the definition of the slope: nudge by $h$, divide the change by $h$, then imagine $h$ getting smaller and smaller. You never carry out an infinite process; the notation names the number the shrinking ratios settle toward. In lesson 0001 the nudge $h = 0.001$ gives a slope of -34.9925, a smaller $h$ would give something closer to -35, and the limit is exactly -35.

## The left arrow: $w \leftarrow w - 0.05 \cdot g$

An arrow pointing left means assignment, exactly like `=` in python: compute the right-hand side using the current value of $w$, then store the result back into $w$. Mathematicians use the arrow because the equals sign already has a job (stating that two things are the same, permanently), and $w = w - 0.05 \cdot g$ would be a false statement rather than an instruction.

## The dot: $a \cdot b$

A centered dot is multiplication. Notation drops the symbol entirely when it can, so $wx$ means $w \cdot x$ means `w * x`.

## Greek letters you will meet

Only a few, and each is a plain variable that happens to be Greek: $\eta$ (eta) sometimes names the learning rate in other books (this repo writes $\mathrm{lr}$ instead), $\sigma$ (small sigma) will name an activation function in lesson 0002, and $\Sigma$ (capital sigma) is the summation above. Greek letters carry no special powers; they exist because the Latin alphabet ran out.
