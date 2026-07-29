# The computation graph

Reference: [directed acyclic graph on Wikipedia](https://en.wikipedia.org/wiki/Directed_acyclic_graph).

A computation graph is what a formula looks like when you draw it as wiring instead of writing it as a line. Every operation becomes a box, every intermediate result becomes a wire, and the picture makes one thing obvious that the written formula hides: the [backward pass](autodiff.md) is a walk through the boxes in reverse. This is the data structure lesson 0004 builds, and the reason an autograd engine is short enough to read in one sitting.

## A formula, drawn

Take lesson 0001's loss on one point, $x = 2$, $y = 5$, at $w = 0$, $b = 0$. Written out it is $L = (w x + b - y)^2$. Built up one operation at a time it is four steps, and each step is a box fed by the results before it:

```
w ---\
      (mul) --> wx ---\
x ---/                 (add) --> z ---\
              b ------/                (sub) --> e --> (square) --> L
                             y -------/
```

The boxes on the left with nothing feeding them are the leaves, the inputs $w, x, b, y$. Every other box is an operation. Reading left to right and filling in numbers is the forward pass: $wx = 0$, $z = 0$, $e = -5$, $L = 25$. The graph is a [directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph): arrows have a direction (inputs into operations), and nothing loops back on itself, which is what lets the backward walk finish.

## Every box carries one local slope

Each box knows a single fact: how much its output moves when its input is nudged by a hair. That is the box's local slope, and it comes from the [derivative](derivative.md) table you already have.

| Box | Output | Local slope back to each input |
|-----|--------|-------------------------------|
| multiply | $a \cdot b$ | to $a$ it is $b$, to $b$ it is $a$ |
| add | $a + b$ | 1 to each |
| subtract | $a - b$ | 1 to $a$, $-1$ to $b$ |
| square | $a^2$ | $2a$ |
| relu | $\max(0, a)$ | 1 if $a > 0$, else 0 |

The multiply box is the only one whose slope depends on a value from the forward pass, which is why an engine has each box remember the inputs it was built from.

## The backward walk

To get the slope of $L$ with respect to every leaf, walk the graph from $L$ back toward the leaves, applying the [chain rule](chain-rule.md) at each box: the slope of $L$ with respect to a box's input is the box's local slope times the slope of $L$ with respect to its output, the number arriving from above. Seed the walk with the slope of $L$ with respect to itself, which is 1.

```
seed        dL/dL  = 1
square      dL/de  = 2e * 1      = -10
subtract    dL/dz  = 1  * -10    = -10
add         dL/db  = 1  * -10    = -10
            dL/dwx = 1  * -10    = -10
multiply    dL/dw  = x  * -10    = -20
```

The two leaves that are knobs come out at $\partial L / \partial w = -20$ and $\partial L / \partial b = -10$, the exact numbers the [chain rule page](chain-rule.md) gets for this point by multiplying the ratios in one line. The walk is that line unrolled: each box contributes one factor, and the boxes that looked invisible in the shortcut (the add's silent times 1) are steps in the walk.

## The one rule a hand pass never states: slopes add

When a single value feeds more than one box, each box sends a slope back to it, and the value's true slope is the sum of them all. The smallest case is a value used twice in one expression:

$$y = x + x$$

By algebra $y = 2x$, so the slope is 2. In the graph, $x$ feeds the add box on both wires, the add box returns slope 1 along each, and $x$ collects $1 + 1 = 2$. An engine that overwrote the slope instead of adding would report 1 and be wrong. This is why every backward closure accumulates with `+=` rather than assigns with `=`, and why a training loop must reset every leaf's slope to 0 between steps: a fresh step needs a clean sum, not last step's total carried over. Forgetting that reset is lesson 0004's deliberate failure, and every framework's `zero_grad` is the fix.

## Why it scales

A deep network is a bigger graph, thousands of boxes instead of four, but the walk never changes: seed with 1, and at each box multiply by the local slope and add into the inputs. [Torch's autograd](autodiff.md) records this graph as your forward code runs and walks it for you; this repo's `grad` package is the same thing in about a hundred lines, and lesson 0004 builds its twin in Python. Nothing in the industrial version is an idea you have not already checked here with a calculator.
