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

## The bars: $|x|$

Vertical bars around a number mean its absolute value: the number with any minus sign stripped off, so $|3| = 3$ and $|-3| = 3$. Geometrically it is the distance from zero, which is never negative. Plotted against $x$, the function $y = |x|$ makes a V shape with its corner at the origin, and that V is the whole subject of lesson 0002, since it is the shortest function no straight line can fit.

## One-hot: y = (0, 1, 0)

When the answer is a category rather than a number, the target is written as a list with a 1 in the true class's slot and 0 everywhere else, called a one-hot encoding. With classes numbered 0, 1, 2, the answer "class 1" becomes $y = (0, 1, 0)$. The point of the format is that it speaks the same language as a probability list: a perfect model would output exactly this, all its probability on the truth, so "output minus target" makes sense slot by slot. Lesson 0003's gradient is exactly that subtraction.

## Argmax

$\mathrm{argmax}$ of a list is the position of its largest entry, not the largest entry itself: $\mathrm{argmax}(0.2, 0.7, 0.1) = 1$, because slot 1 holds the biggest value. It is how a classifier's probability list becomes a single verdict when one is demanded. Unlike everything else on this page it has no useful slope (nudging a probability slightly almost never changes which slot wins), which is why training runs on probabilities and [cross-entropy](cross-entropy.md), and argmax only appears after training, at judgment time.

## The conditional bar: $P(b \mid a)$

Read the bar as "given". $P(b \mid a)$ is the probability that the next symbol is $b$ given that the current one is $a$, a single number between 0 and 1. Fix $a$ and let $b$ range over every possible next symbol and you get a whole probability distribution, one row of a table, that sums to 1. Lesson 0005's [bigram model](bigram.md) is nothing but a table of these, one row per current symbol, each row read off by counting.

## The boundary token: $.$

Sequences need to say where they begin and end, so language models reserve one symbol, written as a dot, to mean "edge of a word". Wrapping `cat` as $.\text{cat}.$ lets a single next-symbol table also answer which symbols can start a word (those that follow the dot) and which can end one (those the dot follows). It is punctuation for the model, not for the reader.

## The row lookup: $W[i]$ and $\mathrm{onehot}(i) \cdot W$

Square brackets index into a table: $W[i]$ is row $i$ of a matrix $W$, the whole row of numbers, not a single entry. When $i$ is the index of a symbol, reading $W[i]$ is an [embedding](embedding.md) lookup, and it equals the matrix product $\mathrm{onehot}(i) \cdot W$, because a one-hot row with its single $1$ in position $i$ selects exactly that row and zeroes the rest. Lesson 0006's model is one line in this notation, $\text{logits} = W[\text{current}]$: look up the current symbol's row of scores, then [softmax](softmax.md) it into next-symbol probabilities.

## Greek letters you will meet

Only a few, and each is a plain variable that happens to be Greek: $\eta$ (eta) sometimes names the learning rate in other books (this repo writes $\mathrm{lr}$ instead), $\sigma$ (small sigma) conventionally names an [activation function](relu.md) in other books (the lessons write out names like relu instead), and $\Sigma$ (capital sigma) is the summation above. Greek letters carry no special powers; they exist because the Latin alphabet ran out.
