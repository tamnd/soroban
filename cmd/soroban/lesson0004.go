package main

import (
	"fmt"

	"github.com/tamnd/soroban/grad"
)

// lesson0004 shows the autograd engine directly: it walks the four-box graph of
// one lesson 0001 point, then reproduces the full 0001 and 0002 runs through the
// same grad package. The seven printed lines match lessons/0004-autograd/train.py
// byte for byte, which is the whole point: two languages, one engine, one set of
// numbers. The grad package this calls is the Go twin of that lesson's micrograd.
func lesson0004() {
	// One point from 0001: x=2, y=5, at w=0, b=0. Build the graph, walk it.
	w, x, b, y := grad.New(0), grad.New(2), grad.New(0), grad.New(5)
	wx := w.Mul(x)
	z := wx.Add(b)
	e := z.Sub(y)
	l := e.Sq()
	l.Backward()
	fmt.Println("one point x=2 y=5 w=0 b=0")
	fmt.Printf("forward   wx %.6f  z %.6f  e %.6f  L %.6f\n", wx.Data, z.Data, e.Data, l.Data)
	fmt.Printf("backward  dL/de %.6f  dL/dz %.6f  dL/db %.6f  dL/dw %.6f\n",
		e.Grad, z.Grad, b.Grad, w.Grad)

	l1, dw1, db1, wf, bf := reproduce0001()
	fmt.Printf("0001      step 1 loss %.6f  dw %.6f  db %.6f\n", l1, dw1, db1)
	fmt.Printf("0001      final w %.6f  b %.6f\n", wf, bf)

	l2, g, p := reproduce0002()
	fmt.Printf("0002      step 1 loss %.6f  grads %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n",
		l2, g[0], g[1], g[2], g[3], g[4], g[5], g[6])
	fmt.Printf("0002      final w1 %.6f b1 %.6f w2 %.6f b2 %.6f v1 %.6f v2 %.6f c %.6f\n",
		p[0], p[1], p[2], p[3], p[4], p[5], p[6])
}

// reproduce0001 reruns lesson 0001 through grad and returns step-1 loss and
// gradients plus the final knobs.
func reproduce0001() (l1, dw1, db1, wf, bf float64) {
	xs := []float64{1, 2, 3, 4}
	ys := []float64{3, 5, 7, 9}
	const lr = 0.05
	wv, bv := 0.0, 0.0
	for step := 1; step <= 200; step++ {
		W, B := grad.New(wv), grad.New(bv)
		errs := make([]*grad.Value, len(xs))
		for i := range xs {
			errs[i] = W.Mul(grad.New(xs[i])).Add(B).Sub(grad.New(ys[i])).Sq()
		}
		L := grad.Mean(errs)
		L.Backward()
		if step == 1 {
			l1, dw1, db1 = L.Data, W.Grad, B.Grad
		}
		wv -= lr * W.Grad
		bv -= lr * B.Grad
	}
	return l1, dw1, db1, wv, bv
}

// reproduce0002 reruns lesson 0002 through grad and returns step-1 loss, the
// seven gradients in the same order train.py prints, and the final knobs.
func reproduce0002() (l2 float64, g [7]float64, p [7]float64) {
	xs := []float64{-2, -1, 1, 2}
	ys := []float64{2, 1, 1, 2}
	const lr = 0.1
	// order: w1, b1, w2, b2, v1, v2, c
	k := [7]float64{1, -0.5, -1, -0.5, 1, 1, 0}
	for step := 1; step <= 300; step++ {
		P := [7]*grad.Value{}
		for i := range k {
			P[i] = grad.New(k[i])
		}
		errs := make([]*grad.Value, len(xs))
		for i := range xs {
			xi := grad.New(xs[i])
			h1 := P[0].Mul(xi).Add(P[1]).Relu()
			h2 := P[2].Mul(xi).Add(P[3]).Relu()
			yhat := P[4].Mul(h1).Add(P[5].Mul(h2)).Add(P[6])
			errs[i] = yhat.Sub(grad.New(ys[i])).Sq()
		}
		L := grad.Mean(errs)
		L.Backward()
		if step == 1 {
			l2 = L.Data
			for i := range P {
				g[i] = P[i].Grad
			}
		}
		for i := range k {
			k[i] -= lr * P[i].Grad
		}
	}
	return l2, g, k
}
