# Neural Network Design

Python notes and worked solutions for the textbook *Neural Network Design* by Martin T. Hagan, Howard B. Demuth, Mark Hudson Beale, and Orlando De Jesús.

A free PDF of the textbook is available at [hagan.okstate.edu/NNDesign.pdf](http://hagan.okstate.edu/NNDesign.pdf).

Every script here is a standalone answer to one exercise or worked problem. Run it and it prints the reasoning and the numbers; several also open a matplotlib plot.

## Setup

Requires Python 3.9 or newer.

```bash
git clone https://github.com/jtcass01/Neural-Network-Design.git
```

```bash
cd Neural-Network-Design && pip install -e .
```

The `-e .` install puts the shared `nnd` package on your path, which is what lets each chapter script import the networks. Without it the imports will fail.

## Running a solution

Run any script directly:

```bash
python "Chapter 3 - An Illustrative Example/Exercises/e3_6.py"
```

Scripts named `eX_Y.py` are end-of-chapter exercises, `pX_Y.py` are the book's solved problems, and files under `Examples/` reproduce a worked example from the chapter text.

**Note:** many scripts call `plt.show()`, which opens a plot window and blocks until you close it. If a script appears to hang, look for the window. Close it and the script continues.

## The `nnd` package

The networks and transfer functions are shared across chapters, so they live in one place:

| Module | Contents |
| --- | --- |
| [`nnd/transfer_functions.py`](nnd/transfer_functions.py) | `hardlim`, `hardlims`, `purelin`, `satlin`, `satlins`, `logsig`, `tansig`, `poslin`, `compet` |
| [`nnd/perceptron.py`](nnd/perceptron.py) | `Perceptron` — pages 3-3:3-8, plus the Chapter 4 learning rule |
| [`nnd/hamming.py`](nnd/hamming.py) | `HammingNetwork` — pages 3-8:3-12 |
| [`nnd/hopfield.py`](nnd/hopfield.py) | `HopfieldNetwork` — pages 3-12:3-14 |
| [`nnd/plotting.py`](nnd/plotting.py) | Plot and print helpers used by the Chapter 4 exercises |
| [`nnd/prototypes.py`](nnd/prototypes.py) | `prototype_pairs` — packs (input, target) pairs for training |

`Perceptron` can be built either way the book uses it:

```python
from nnd import Perceptron
from nnd.transfer_functions import hardlims

# Chapter 3: a decision boundary you worked out by hand
network = Perceptron(W=weights, b=bias, transfer_function=hardlims)

# Chapter 4: random starting values, then apply the learning rule
network = Perceptron(number_of_neurons=1, input_size=2)
network.train(prototypes=prototypes)
```

## Chapter 2 — Neuron Model and Network Architectures

### Exercises

| Exercise | Question |
| --- | --- |
| [E2.1](Chapter%202%20-%20Neuron%20Model%20and%20Network%20Architectures/Exercises/e2_1.py) | Which transfer functions could produce each given output? |
| [E2.2](Chapter%202%20-%20Neuron%20Model%20and%20Network%20Architectures/Exercises/e2_2.py) | Single-input neuron with a bias: output -1 below p=3, +1 at or above |
| [E2.3](Chapter%202%20-%20Neuron%20Model%20and%20Network%20Architectures/Exercises/e2_3.py) | Two-input neuron: is there a bias and transfer function giving output 0.5? |
| [E2.4](Chapter%202%20-%20Neuron%20Model%20and%20Network%20Architectures/Exercises/e2_4.py) | Two-layer network, four inputs, six outputs continuous on [0, 1] |
| [E2.5](Chapter%202%20-%20Neuron%20Model%20and%20Network%20Architectures/Exercises/e2_5.py) | Sketch the neuron response over -2 < p < 2 for six weight/bias/transfer combinations |
| [E2.6](Chapter%202%20-%20Neuron%20Model%20and%20Network%20Architectures/Exercises/e2_6.py) | Two-layer network: saturating linear layer into a linear layer |

### Solved Problems

| Problem | Question |
| --- | --- |
| [P2.1](Chapter%202%20-%20Neuron%20Model%20and%20Network%20Architectures/Solved%20Problems/p2_1.py) | Net input to a single-input neuron with p=2.0, w=2.3, b=-3 |
| [P2.2](Chapter%202%20-%20Neuron%20Model%20and%20Network%20Architectures/Solved%20Problems/p2_2.py) | Output of that neuron under hard limit, linear, and log-sigmoid |
| [P2.3](Chapter%202%20-%20Neuron%20Model%20and%20Network%20Architectures/Solved%20Problems/p2_3.py) | Two-input neuron output under four different transfer functions |
| [P2.4](Chapter%202%20-%20Neuron%20Model%20and%20Network%20Architectures/Solved%20Problems/p2_4.py) | Single-layer network, six inputs, two outputs continuous on [0, 1] |

## Chapter 3 — An Illustrative Example

### Examples

| Example | Contents |
| --- | --- |
| [Perceptron](Chapter%203%20-%20An%20Illustrative%20Example/Examples/Perceptron.py) | Apple/orange classifier, pages 3-3:3-8 |
| [Hamming network](Chapter%203%20-%20An%20Illustrative%20Example/Examples/HammingNetwork.py) | Apple/orange classifier, pages 3-8:3-12 |
| [Hopfield network](Chapter%203%20-%20An%20Illustrative%20Example/Examples/HopfieldNetwork.py) | Apple/orange classifier, pages 3-12:3-14 |

### Exercises

| Exercise | Question |
| --- | --- |
| [E3.1](Chapter%203%20-%20An%20Illustrative%20Example/Exercises/e3_1.py) | Redesign all three networks to tell bananas from pineapples |
| [E3.2](Chapter%203%20-%20An%20Illustrative%20Example/Exercises/e3_2.py) | Decision boundary, weights, and bias for two prototype patterns |
| [E3.3](Chapter%203%20-%20An%20Illustrative%20Example/Exercises/e3_3.py) | Hopfield network with a given weight and bias |
| [E3.4](Chapter%203%20-%20An%20Illustrative%20Example/Exercises/e3_4.py) | Analyze a given perceptron network |
| [E3.5](Chapter%203%20-%20An%20Illustrative%20Example/Exercises/e3_5.py) | Perceptron outputting 1 for two vectors and -1 for two others |
| [E3.6](Chapter%203%20-%20An%20Illustrative%20Example/Exercises/e3_6.py) | Two prototype vectors through all three network types |
| [E3.7](Chapter%203%20-%20An%20Illustrative%20Example/Exercises/e3_7.py) | Hamming network to recognize given prototype vectors |

## Chapter 4 — Perceptron Learning Rule

### Examples

| Example | Contents |
| --- | --- |
| [Perceptron](Chapter%204%20-%20Perceptron%20Learning%20Rule/Examples/Perceptron.py) | Training from random weights, on a 3-input problem and an AND gate |
| [Decision boundary](Chapter%204%20-%20Perceptron%20Learning%20Rule/Examples/decision_boundary.py) | AND gate: decision boundary, weights, and bias |

### Exercises

| Exercise | Question |
| --- | --- |
| [E4.1](Chapter%204%20-%20Perceptron%20Learning%20Rule/Exercises/e4_1.py) | Five-point classification problem: is it linearly separable? |
| [E4.2](Chapter%204%20-%20Perceptron%20Learning%20Rule/Exercises/e4_2.py) | Design a single-neuron perceptron graphically, then classify four new vectors |

## Remaining chapters

Chapters 2, 3, and 4 are done in Python here. I will come back to this book when I have time.

In the meantime, [@estamos](https://github.com/estamos/Neural-Network-Design-Solutions-Manual) has completed additional chapters in MATLAB — [10](https://github.com/estamos/Neural-Network-Design-Solutions-Manual/tree/master/Chapter%2010), [11](https://github.com/estamos/Neural-Network-Design-Solutions-Manual/tree/master/Chapter%2011), [12](https://github.com/estamos/Neural-Network-Design-Solutions-Manual/tree/master/Chapter%2012), [16](https://github.com/estamos/Neural-Network-Design-Solutions-Manual/tree/master/Chapter%2016), and [17](https://github.com/estamos/Neural-Network-Design-Solutions-Manual/tree/master/Chapter%2017). The PDFs are written in Greek, but Google Translate handles them.

## About the authors of the textbook

- **Martin T. Hagan** — Ph.D. Electrical Engineering, University of Kansas; Professor in the School of Electrical and Computer Engineering at Oklahoma State University
- **Howard B. Demuth** — Ph.D. Electrical Engineering, Stanford University; teaches a neural network course at the University of Colorado at Boulder
- **Mark Hudson Beale** — B.S. Computer Engineering, University of Idaho
- **Orlando De Jesús** — Ph.D. Electrical Engineering, Oklahoma State University

## License

[GNU General Public License v3.0](LICENSE)
