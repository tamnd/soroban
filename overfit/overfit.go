// Package overfit is lesson 0008's by-hand core: the arithmetic that shows why the
// training loss alone cannot tell you whether a model has learned or merely
// memorized. Fit a small set of points with a straight line (two parameters) and
// again with a parabola (three parameters, one per point). The parabola drives the
// training error to exactly zero by bending through every point, including the
// noise, while the line keeps a small training error and a much smaller error on
// points it never saw. That gap between fitting the training data and predicting
// held-out data is overfitting, and this package computes it in closed form, plus
// the parameter count of lesson 0007's transformer, so the Go and Python headlines
// diff empty.
package overfit

// FitPoly returns the least-squares polynomial of the given degree through the
// points (xs, ys), as coefficients in ascending powers: c[0] + c[1]*x + ... . With
// degree equal to the number of points minus one the fit is exact (it interpolates,
// training error zero); with a lower degree it is the best straight-ish fit and
// generally leaves a residual. The coefficients solve the normal equations
// A^T A c = A^T y, where A is the Vandermonde matrix of the xs.
func FitPoly(xs, ys []float64, degree int) []float64 {
	n := len(xs)
	cols := degree + 1
	// Vandermonde: a[i][p] = xs[i]^p.
	a := make([][]float64, n)
	for i := range n {
		a[i] = make([]float64, cols)
		v := 1.0
		for p := range cols {
			a[i][p] = v
			v *= xs[i]
		}
	}
	// Normal equations: ata = A^T A (cols x cols), aty = A^T y (cols).
	ata := make([][]float64, cols)
	aty := make([]float64, cols)
	for r := range cols {
		ata[r] = make([]float64, cols)
		for i := range n {
			aty[r] += a[i][r] * ys[i]
			for c := range cols {
				ata[r][c] += a[i][r] * a[i][c]
			}
		}
	}
	return solve(ata, aty)
}

// solve returns x for the square system A x = b by Gaussian elimination with
// partial pivoting. The systems here are tiny (at most three by three).
func solve(a [][]float64, b []float64) []float64 {
	n := len(b)
	// Work on copies so the caller's matrices are untouched.
	m := make([][]float64, n)
	for i := range n {
		m[i] = make([]float64, n+1)
		copy(m[i], a[i])
		m[i][n] = b[i]
	}
	for col := range n {
		// Pivot on the largest magnitude entry in this column.
		piv := col
		for r := col + 1; r < n; r++ {
			if abs(m[r][col]) > abs(m[piv][col]) {
				piv = r
			}
		}
		m[col], m[piv] = m[piv], m[col]
		// Eliminate below.
		for r := col + 1; r < n; r++ {
			f := m[r][col] / m[col][col]
			for c := col; c <= n; c++ {
				m[r][c] -= f * m[col][c]
			}
		}
	}
	// Back-substitute.
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		s := m[i][n]
		for c := i + 1; c < n; c++ {
			s -= m[i][c] * x[c]
		}
		x[i] = s / m[i][i]
	}
	return x
}

// Eval evaluates a polynomial (ascending-power coefficients) at x by Horner's rule.
func Eval(coef []float64, x float64) float64 {
	y := 0.0
	for i := len(coef) - 1; i >= 0; i-- {
		y = y*x + coef[i]
	}
	return y
}

// MSE returns the mean squared error of a polynomial fit over the points (xs, ys).
func MSE(coef, xs, ys []float64) float64 {
	sum := 0.0
	for i := range xs {
		e := Eval(coef, xs[i]) - ys[i]
		sum += e * e
	}
	return sum / float64(len(xs))
}

// ParamCount returns the number of trained parameters in lesson 0007's one-block,
// one-head transformer for vocabulary V, width D, and context length T: token and
// position embeddings, the four attention projections, two feed-forward layers, the
// three layer norms, and the output head. The same architecture, counted the same
// way, that trains on TinyStories in lesson 0007 and is measured here in lesson 0008.
func ParamCount(V, D, T int) int {
	tok := V * D                       // token embedding
	pos := T * D                       // position embedding
	qkv := 3 * (D * D)                 // query, key, value projections, no bias
	proj := D*D + D                    // output projection of the head, with bias
	mlp := (D*4*D + 4*D) + (4*D*D + D) // two feed-forward layers, both with bias
	norms := 3 * (2 * D)               // three layer norms, each a scale and a shift
	head := D*V + V                    // linear map to vocabulary logits, with bias
	return tok + pos + qkv + proj + mlp + norms + head
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
