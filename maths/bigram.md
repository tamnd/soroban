# The bigram model, language as counting

Reference: [n-gram on Wikipedia](https://en.wikipedia.org/wiki/N-gram).

A bigram model is the smallest thing that deserves to be called a language model. It assumes the next symbol in a sequence depends only on the current one, and it estimates that dependence by counting: go through the training text, tally how often each symbol follows each other symbol, and turn the tallies into probabilities. No gradient, no weights to learn by descent, just a table of counts and a division. Everything harder in this repo is a way of relaxing the one crude assumption, that only the current symbol matters, but the loss, the sampling, and the failure modes all show up first here where they can be checked by hand.

## The boundary token

Words have to start and stop somewhere, so wrap every word in a single boundary symbol, written `.`, before counting: `cat` becomes `.cat.`. The leading `.` is the state the model is in before it has written anything, and the trailing `.` is how it signals it is done. This one trick lets the same count table answer three questions at once: which symbols start a word (what follows `.`), which symbols end one (what precedes `.`), and which follow which in the middle. A model without a boundary token can predict the next letter but never decide to stop.

## Counting into probabilities

Let $N(a, b)$ be the number of times symbol $b$ followed symbol $a$. The model's probability for the next symbol is the fraction of times that symbol actually came next:

$$P(b \mid a) = \frac{N(a, b)}{\sum_{b'} N(a, b')}$$

the count of the pair over the total count of everything that followed $a$. This is the maximum likelihood estimate: among all possible probability tables, this is the one that makes the training text most probable, and it turns out to be exactly the observed frequencies. Each row of the table (fix $a$, vary $b$) is a distribution: non-negative and summing to 1, the same shape a [softmax](softmax.md) produces, but obtained by counting instead of from scores.

Worked on lesson 0005's corpus `cat cot cab`, the letter `c` is followed by `a` twice and `o` once, so $P(a \mid c) = 2/3$ and $P(o \mid c) = 1/3$. The letter `a` is followed by `t` once and `b` once, so each is $1/2$. Every count is small enough to verify by sliding a two-symbol window along `.cat. .cot. .cab.` and tallying.

## The loss is cross-entropy on a table

The probability a model assigns to a whole word is the product of its bigram probabilities along the word, boundaries included:

$$P(\text{word}) = \prod_{\text{pairs } (a,b) \text{ in word}} P(b \mid a)$$

A product of small numbers is awkward to work with and underflows fast, so take the negative logarithm, which turns the product into a sum and the "high probability is good" goal into "low number is good". The loss over a corpus is the average of $-\ln P(b \mid a)$ across every bigram:

$$L = -\frac{1}{M} \sum_{\text{all } M \text{ bigrams}} \ln P(b \mid a)$$

Each term is the [cross-entropy](cross-entropy.md) loss from lesson 0003 with a one-hot target: the true next symbol has target 1, the rest 0, and the term is $-\ln$ of the probability placed on the true one. A bigram model just reads that probability off the count table instead of computing it from a softmax, so the loss carries over unchanged. On `cat cot cab` every word has probability exactly $1/3$, so the loss is $\ln(3)/4 = 0.274653$ (four bigrams per word, each word contributing $\ln 3$, averaged over twelve bigrams). The number a model that memorized three equiprobable words should report, checkable on paper.

## Sampling: reading the table backward

A trained table can be run as a generator. Start in state `.`, look at that row of probabilities, draw the next symbol at random weighted by those probabilities, move to it, and repeat until the draw lands on `.`. The word written along the way is a sample from the model. To draw one symbol from a row by hand, use inverse-transform sampling: line the symbols up in a fixed order, take a roll $u$ between 0 and 1, and walk the running total of probabilities until it first reaches $u$; the symbol you stopped on is the draw. A roll of $0.2$ against the row for `c` (running totals: `a` reaches $0.667$, then `o` reaches $1.0$) stops at `a`. Feeding a fixed list of rolls makes sampling fully deterministic, which is how the same words come out in numpy, torch, and Go.

## The zero-frequency problem

Pure counting has a sharp failure: any bigram it never saw gets count 0, so probability 0, so $-\ln 0 = \infty$. Score a word whose first letter never started a word in the training set and the model does not just doubt it, it calls it impossible, and one impossible word makes the loss on any dataset containing it infinite. Real corpora never contain every valid pair, so this is not an edge case, it is the normal state of a counting model on held-out text.

The first fix is add-one (Laplace) smoothing: pretend every possible bigram was seen one extra time before counting. With an alphabet of $V$ symbols,

$$P(b \mid a) = \frac{N(a, b) + 1}{\left(\sum_{b'} N(a, b')\right) + V}$$

Now no probability is ever zero, an unseen word gets a small but finite loss, and the cost is a slight blurring of the probabilities the model was confident about. Smoothing is the counting model's version of a model admitting it has not seen everything, and the neural language models later in the ladder get the same effect for free, because a softmax over real-valued scores can never output an exact zero.
