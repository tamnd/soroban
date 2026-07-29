// Package grad implements reverse-mode automatic differentiation on scalars.
//
// A Value wraps a float64 and remembers how it was computed, so that after
// building an expression like loss = mean((w*x + b - y)^2) you can call
// loss.Backward() and read the derivative of loss with respect to every
// input out of the Grad fields. This is the same trick torch.autograd does,
// shrunk to the point where you can read all of it in one sitting.
package grad

// Value is one node in the expression graph: a number, the gradient of the
// final output with respect to that number, and a closure that knows how to
// push gradient back to the nodes it was computed from.
type Value struct {
	Data float64
	Grad float64
	prev []*Value
	back func()
}

// New returns a leaf Value with no history.
func New(x float64) *Value {
	return &Value{Data: x}
}

// Add returns a + b.
func (a *Value) Add(b *Value) *Value {
	out := &Value{Data: a.Data + b.Data, prev: []*Value{a, b}}
	out.back = func() {
		a.Grad += out.Grad
		b.Grad += out.Grad
	}
	return out
}

// Sub returns a - b.
func (a *Value) Sub(b *Value) *Value {
	out := &Value{Data: a.Data - b.Data, prev: []*Value{a, b}}
	out.back = func() {
		a.Grad += out.Grad
		b.Grad -= out.Grad
	}
	return out
}

// Mul returns a * b. The backward rule is the product rule: each factor
// receives the incoming gradient scaled by the other factor's value.
func (a *Value) Mul(b *Value) *Value {
	out := &Value{Data: a.Data * b.Data, prev: []*Value{a, b}}
	out.back = func() {
		a.Grad += b.Data * out.Grad
		b.Grad += a.Data * out.Grad
	}
	return out
}

// Sq returns a squared. The slope of x^2 is 2x, which is exactly the fact
// lesson 0001 checks by nudging a number on a calculator.
func (a *Value) Sq() *Value {
	out := &Value{Data: a.Data * a.Data, prev: []*Value{a}}
	out.back = func() {
		a.Grad += 2 * a.Data * out.Grad
	}
	return out
}

// Mean returns the average of vs.
func Mean(vs []*Value) *Value {
	sum := vs[0]
	for _, v := range vs[1:] {
		sum = sum.Add(v)
	}
	return sum.Mul(New(1 / float64(len(vs))))
}

// Backward runs backpropagation from v: it topologically sorts the graph,
// seeds v.Grad = 1, and applies every node's chain rule in reverse order.
// Gradients accumulate, so zero them between calls if you reuse leaves.
func (v *Value) Backward() {
	var topo []*Value
	visited := map[*Value]bool{}
	var build func(*Value)
	build = func(n *Value) {
		if visited[n] {
			return
		}
		visited[n] = true
		for _, p := range n.prev {
			build(p)
		}
		topo = append(topo, n)
	}
	build(v)

	v.Grad = 1
	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i].back != nil {
			topo[i].back()
		}
	}
}
