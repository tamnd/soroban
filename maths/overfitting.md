# Overfitting, when fitting the data stops meaning learning it

Reference: [overfitting on Wikipedia](https://en.wikipedia.org/wiki/Overfitting).

Every lesson so far judged a model by one number: the loss on the data it trained on. Drive that number down, the story went, and the model has learned. This page is where that story breaks. A model with enough free parameters can drive its training loss to zero by memorizing the training data point for point, and a memorized answer key tells you nothing about a question the model has not seen. The fix is to hold some data back, never train on it, and measure the loss there too. The gap between the two losses is the thing to watch, and it has a name: overfitting.

## Capacity, and the point where fitting becomes memorizing

A model's capacity is, loosely, how many independent numbers it can bend to the data, which is its parameter count. The sharpest way to see capacity at work is polynomial fitting, where the parameter count is the degree plus one. A straight line $y = ax + b$ has two parameters. A parabola $y = ax^2 + bx + c$ has three. The rule that matters: a polynomial with as many parameters as there are data points can pass exactly through all of them, driving the training error to zero, no matter what the data is. Two parameters fix a line through any two points; three parameters fix a parabola through any three. When the parameter count reaches the number of data points, the model can memorize, and its training error stops being evidence of anything.

## The worked example

Take a true relationship, the line $y = 0.5x + 0.5$, and sample it at $x = 0, 1, 2$, but nudge the middle sample off the line: instead of its true value $1.0$ use $1.3$. That nudge is noise, the kind of small corruption real data always carries. The three training points are

$$(0,\ 0.5), \qquad (1,\ 1.3), \qquad (2,\ 1.5).$$

Hold out two clean points that sit exactly on the true line, at $x = 3$ and $x = 4$, so $y = 2.0$ and $y = 2.5$. Now fit two models.

The least-squares line through the three training points comes out to $y = 0.5x + 0.6$. It cannot pass through all three, because three generic points are not collinear, so it keeps a training error. Its predictions are $0.6, 1.1, 1.6$ against targets $0.5, 1.3, 1.5$, for squared errors $0.01, 0.04, 0.01$ and a training mean squared error of

$$\text{MSE}_{\text{train}}^{\text{line}} = \frac{0.01 + 0.04 + 0.01}{3} = 0.02.$$

The parabola has three parameters for three points, so it interpolates them exactly: $y = -0.3x^2 + 1.1x + 0.5$ passes through all three, with a training error of exactly zero. Check one, at $x = 1$: $-0.3 + 1.1 + 0.5 = 1.3$, the noisy point, hit dead on. On the training set the parabola wins outright, $0$ against $0.02$.

## The held-out set changes the verdict

Read the two models on the points they never saw. The line at $x = 3, 4$ predicts $2.1, 2.6$ against the true $2.0, 2.5$, squared errors $0.01, 0.01$, for a held-out MSE of $0.01$. The parabola, which bent downward ($-0.3x^2$) to catch the noisy middle point, keeps bending: at $x = 3$ it predicts $1.1$ and at $x = 4$ it predicts $0.1$, against the true $2.0$ and $2.5$. Its squared errors are $0.81$ and $5.76$, for a held-out MSE of

$$\text{MSE}_{\text{val}}^{\text{parab}} = \frac{0.81 + 5.76}{2} = 3.285.$$

Line the four numbers up.

| model | parameters | train MSE | held-out MSE |
|-------|-----------:|----------:|-------------:|
| line | 2 | 0.020 | 0.010 |
| parabola | 3 | 0.000 | 3.285 |

On the training set the parabola looks perfect and the line looks worse. On the held-out set the line is 328 times better. If you chose your model by training error alone, as every earlier lesson implicitly did, you would pick the parabola, the worse model, every time. The held-out set is the instrument that catches the mistake, and the gap between a model's training error and its held-out error is the reading that instrument gives.

## The train/validation split

The recipe generalizes past polynomials. Before training anything, split the data into a training set the model learns from and a validation set it never touches. Train on the first, and periodically measure the loss on the second. Two curves come out of a training run: training loss, which almost always keeps falling, and validation loss, which falls at first, because early on the model is learning real structure that helps everywhere, and then, once the model starts fitting quirks specific to the training set, turns and climbs. That U-shape is overfitting in motion. The bottom of the U is the moment the model knew the most that transfers, and stopping there, called early stopping, is the simplest defense. Training past it trades generalization for a lower training number that means nothing.

## What decides how bad it gets

The size of the gap tracks the ratio of parameters to data points. Below one parameter per data point the model lacks the capacity to memorize and the two losses stay close; the risk there is the opposite, underfitting, too little capacity to capture even the real pattern. Around and above one parameter per data point the model can start memorizing and the gap opens. Lesson 0007's transformer has 58273 parameters and trained on 9311 characters, about six parameters per character, well into the regime where memorizing is not just possible but easy, which is exactly why its training loss fell so far. Three levers close the gap: fewer parameters, more data, or stopping early. Lesson 0008 turns the validation instrument on that transformer and reads all three off the curves.
