# soroban

[![ci](https://github.com/tamnd/soroban/actions/workflows/ci.yml/badge.svg)](https://github.com/tamnd/soroban/actions/workflows/ci.yml)

> [!NOTE]
> The GPU lessons in this repo are developed and tested on a real RTX 4090 machine generously provided by [longkt90](https://github.com/longkt90). None of the hardware measurements would exist without it. Thank you.

Soroban (算盤) is the Japanese abacus, the tool for doing real arithmetic by hand. That is the whole idea of this repo: learn to train neural networks by computing the training runs yourself, on paper, with a four-function calculator, and only then letting the machines confirm your numbers.

I started this because every explanation of backpropagation I read either hid the arithmetic behind matrix notation or hid it behind a framework. Neither ever made it stick for me. What made it stick was sitting down and computing three steps of gradient descent by hand, then writing code whose only job was to agree with my paper. So that is the format of every lesson here.

Every lesson follows the same discipline. You predict what will happen, you compute a small training run entirely by hand, then code reproduces your arithmetic with `assert` statements standing guard over every number, and finally you break something on purpose and study the wreckage. If the asserts pass, the machine has agreed with your paper. If they fail, you get to find out who is wrong, and learning to find out who is wrong is the actual curriculum.

There is no required background beyond arithmetic. The one calculus idea we need (a derivative is a slope you can measure by nudging) is built from a numerical experiment inside lesson 0001, not imported from a course you were supposed to have taken. And when notation gets in the way, the [maths shelf](maths) has one plain-language page per symbol and idea, each worked out on the same numbers the lessons use, so the lessons can stay focused on training while the shelf carries the reference material.

## The lessons

| # | Lesson | You compute by hand | Runs on | Status |
|---|--------|--------------------|---------|--------|
| 0001 | [one neuron learns a line](lessons/0001-one-neuron) | 3 full gradient descent steps, every multiplication shown | any CPU | done |
| 0002 | [a hidden layer learns a V](lessons/0002-hidden-layer) | a full backward pass through two layers, all seven gradients | any CPU | done |
| 0003 | [ice, water, steam](lessons/0003-classification) | softmax and cross-entropy on a 3-class example | any CPU | done |
| 0004 | [autograd from scratch](lessons/0004-autograd) | the computation graph of 0001, walked backward box by box | any CPU | done |
| 0005 | [language as counting](lessons/0005-bigram) | a bigram table over a tiny corpus, sampling included | any CPU | done |
| 0006 | embeddings | one embedding lookup and its gradient | any CPU | planned |
| 0007 | attention | one attention head over 3 tokens, every dot product | GPU helps | planned |
| 0008 | the training loop as instrument | parameters vs data points, overfitting arithmetic | GPU helps | planned |
| 0009 | the FLOPs ledger | the 6ND rule for GPT-2, on paper, then measured | GPU (4090) | planned |
| 0010 | fine-tuning arithmetic | LoRA parameter counting | GPU (4090) | planned |

Each lesson folder contains a `README.md` written as a blog post that explains every concept as it appears, a `lesson.ipynb` notebook that runs locally or on Google Colab with one click, a `train.py` script for real hardware, and a `visuals.py` of [manim](https://www.manim.community/) scenes that render the animations and figures embedded in the writeup. The pictures are not decoration; each one is generated from the same numbers the asserts pin down.

## Running things

**Locally.** Install [uv](https://docs.astral.sh/uv/) and you are done setting up; `uv run lessons/0001-one-neuron/train.py` fetches what it needs and runs. Plain python with numpy also works.

**On Google Colab.** Every lesson README carries an "Open in Colab" badge for its notebook. Colab preinstalls numpy, torch, and matplotlib, so the notebooks run with zero setup on the free tier; nothing before lesson 0007 needs more than a CPU.

**On a real GPU box.** Same scripts. Lessons 0001 through 0006 will not touch the GPU and that is deliberate, the algorithm is hardware-blind and you should see that with your own eyes. From 0007 the models get big enough to feed a GPU, and 0009 and 0010 are specifically about measuring one.

## The Go library

The repo doubles as a small Go library, built from scratch and grown one lesson at a time:

- [`grad`](grad): a reverse-mode autograd engine on scalars, about a hundred readable lines. The same machinery as `torch.autograd`, minus everything that is not the idea itself.
- [`nn`](nn): neurons and, as the lessons demand them, layers and networks on top of `grad`.
- [`cmd/soroban`](cmd/soroban): one runner for all lessons. `go run ./cmd/soroban list` shows what exists, `go run ./cmd/soroban 0001` trains lesson 0001 and prints the same table as its `train.py`, byte for byte.

The Go tests assert the same hand-computed numbers as the python (`go test ./...`), so every lesson's arithmetic ends up pinned by three independent implementations: your paper, numpy, and Go. When two of them agree and one does not, you know where the bug lives. That cross-check is the best debugging trick this repo has to teach, and it falls out of the structure for free. CI runs all three and also diffs the python table against the Go table, so the agreement is enforced on every pull request, not just promised.

## Layout

```
soroban/
├── grad/               autograd engine (from scratch, scalar-valued)
├── nn/                 neural net pieces built on grad
├── cmd/soroban/        lesson runner: list lessons, run one by id
├── maths/              the maths shelf: notation and concepts, one page each
├── lessons/
│   └── 0001-one-neuron/
│       ├── README.md   the lesson, written to be read
│       ├── lesson.ipynb  the lesson, written to be run (local or Colab)
│       ├── train.py    the lesson, for real hardware
│       ├── visuals.py  manim scenes for the figures
│       └── assets/     rendered animations and figures
├── pyproject.toml      python deps, uv-friendly
└── .github/workflows/  ci runs the asserts in both languages, plus the cross-check
```

## License

MIT. See [LICENSE](LICENSE).
