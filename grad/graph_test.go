package grad

import "testing"

// These are lesson 0004's by-hand numbers: the four-box graph of one 0001 point
// walked backward, and the two reused-node cases the lesson uses to show why
// gradients must accumulate. Every value is binary-exact, so equality is exact.

// TestOnePointGraph walks the graph of one 0001 point (x=2, y=5, w=0, b=0) and
// checks every intermediate slope against 01-by-hand section 3.
func TestOnePointGraph(t *testing.T) {
	w, x, b, y := New(0), New(2), New(0), New(5)
	wx := w.Mul(x)
	z := wx.Add(b)
	e := z.Sub(y)
	l := e.Sq()
	l.Backward()

	forward := []struct {
		name string
		got  float64
		want float64
	}{
		{"wx", wx.Data, 0}, {"z", z.Data, 0}, {"e", e.Data, -5}, {"L", l.Data, 25},
	}
	for _, c := range forward {
		if c.got != c.want {
			t.Errorf("forward %s: got %v, want %v", c.name, c.got, c.want)
		}
	}

	backward := []struct {
		name string
		got  float64
		want float64
	}{
		{"dL/de", e.Grad, -10}, {"dL/dz", z.Grad, -10},
		{"dL/db", b.Grad, -10}, {"dL/dwx", wx.Grad, -10},
		{"dL/dw", w.Grad, -20}, {"dL/dx", x.Grad, 0},
	}
	for _, c := range backward {
		if c.got != c.want {
			t.Errorf("backward %s: got %v, want %v", c.name, c.got, c.want)
		}
	}
}

// TestReusedNode is the smallest test that catches a missing accumulation: a
// value routed into both inputs of one operation. x+x has slope 2, x*x has
// slope 2x. An engine that overwrote instead of adding would read 1 and x.
func TestReusedNode(t *testing.T) {
	x := New(3)
	sum := x.Add(x)
	sum.Backward()
	if sum.Data != 6 || x.Grad != 2 {
		t.Errorf("x+x: got value %v grad %v, want 6 and 2", sum.Data, x.Grad)
	}

	x = New(3)
	prod := x.Mul(x)
	prod.Backward()
	if prod.Data != 9 || x.Grad != 6 {
		t.Errorf("x*x: got value %v grad %v, want 9 and 6", prod.Data, x.Grad)
	}
}
