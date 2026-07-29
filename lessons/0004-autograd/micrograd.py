"""A reverse-mode autograd engine on scalars, built from nothing.

A Value wraps a float and remembers how it was computed, so that after
building an expression like loss = mean((w*x + b - y)**2) you can call
loss.backward() and read the derivative of loss with respect to every input
out of the .grad fields. This is the same trick torch.autograd does, and the
exact twin of this repo's Go grad package, shrunk to the point where you can
read all of it in one sitting.

The whole engine is below. Lesson 0004 builds it and proves it reproduces
lessons 0001 and 0002 to the digit; see train.py.
"""

import math


class Value:
    """One node in the expression graph: a number, the gradient of the final
    output with respect to that number, the nodes it was computed from, and a
    closure that pushes gradient back to those nodes."""

    def __init__(self, data, prev=()):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(prev)
        self._back = lambda: None  # leaves have nothing to push to

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def back():
            # add sends its incoming gradient to both inputs unchanged
            self.grad += out.grad
            other.grad += out.grad

        out._back = back
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other))

        def back():
            self.grad += out.grad
            other.grad -= out.grad

        out._back = back
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def back():
            # the product rule: each factor scaled by the other's value
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._back = back
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other))

        def back():
            # the quotient rule, split: 1/b for the numerator, -a/b^2 for b
            self.grad += out.grad / other.data
            other.grad -= self.data / (other.data * other.data) * out.grad

        out._back = back
        return out

    def sq(self):
        out = Value(self.data * self.data, (self,))

        def back():
            # the slope of x^2 is 2x, the fact lesson 0001 checks by nudging
            self.grad += 2 * self.data * out.grad

        out._back = back
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,))

        def back():
            # a gate: gradient flows only through inputs that fired
            if self.data > 0:
                self.grad += out.grad

        out._back = back
        return out

    def exp(self):
        d = math.exp(self.data)
        out = Value(d, (self,))

        def back():
            # e^x is its own slope, so the backward rule reuses the forward
            self.grad += d * out.grad

        out._back = back
        return out

    def log(self):
        out = Value(math.log(self.data), (self,))

        def back():
            # the slope of ln(x) is 1/x, largest where x is smallest
            self.grad += out.grad / self.data

        out._back = back
        return out

    # a * scalar and scalar * a, so 0.25 * loss reads naturally
    __radd__ = __add__
    __rmul__ = __mul__

    def backward(self):
        """Topologically sort the graph, seed this node's grad to 1, and apply
        every node's chain rule in reverse order. Gradients accumulate, so zero
        the leaves between calls if you reuse them."""
        topo, visited = [], set()

        def build(n):
            if id(n) in visited:
                return
            visited.add(id(n))
            for p in n._prev:
                build(p)
            topo.append(n)

        build(self)
        self.grad = 1.0
        for n in reversed(topo):
            n._back()


def mean(vs):
    """The average of a list of Values, as one more node on the graph."""
    total = vs[0]
    for v in vs[1:]:
        total = total + v
    return total * (1.0 / len(vs))
