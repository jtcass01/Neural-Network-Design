
"""
We want to design a Hamming network to recognize the following prototype vectors:
    p1 = [1, 1].T   p2 = [-1, -1].T     p3 = [-1, 1].T

i.   Find the weight matrices and bias vectors for the Hamming network.
ii.  Draw the network diagram.
iii. Apply the following input vector and calculate the total network response
    (iterating the second layer to convergence). Explain the meaning of the final network output.
    p_test = [1, 0]
iv.  Sketch the decision boundaries for this network. Explain how you determined the boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
from HammingNetwork import HammingNetwork

if __name__ == "__main__":
    print("We want to design a Hamming network to recognize the following prototype vectors:",
          "\n\tp1 = [1, 1].T", "p2 = [-1, -1].T", "p3 = [-1, 1].T")
    p1 = np.array([1, 1]).reshape((2, 1))
    p2 = np.array([-1,-1]).reshape((2, 1))
    p3 = np.array([-1, 1]).reshape((2, 1))

    prototypes = np.array([p1.T, p2.T, p3.T]).reshape((3, 2))
    print("prototypes", prototypes)

    e3_7_hamming = HammingNetwork(prototypes=prototypes)

    print("p1", p1, "classification:", e3_7_hamming.classify(p1))
    print("p2", p2, "classification:", e3_7_hamming.classify(p2))
    print("p3", p3, "classification:", e3_7_hamming.classify(p3))

    print("\ni.   Find the weight matrices and bias vectors for the Hamming network.",
          "\n\tFeedForwardLayer", "Weights", e3_7_hamming.feedForwardLayer.Weights, "bias", e3_7_hamming.feedForwardLayer.bias,
          "\n\tRecurrentLayer", "Weights", e3_7_hamming.recurrentLayer.Weights, "bias", "There are no biases in the recurrrent layer.")

    print("\niii. Apply the following input vector and calculate the total network response (iterating the second layer to convergence). Explain the meaning of the final network output.",
          "\n\tp_test = [1, 0].T")
    p_test = np.array([1, 0]).reshape((2, 1))
    print("\tClassification:", e3_7_hamming.classify(p_test))

    print("\niv.  Sketch the decision boundaries for this network. Explain how you determined the boundaries.")
    x = np.linspace(-2, 2, 100)
    y = np.linspace(0, 0, 100)

    x2 = np.linspace(0, 0, 100)
    y2 = np.linspace(-2, 2, 100)

    plt.scatter(prototypes[:, 0], prototypes[:, 1])
    plt.plot(x, y, color='black')
    plt.plot(x2, y2, color='black')
    plt.grid(True)
    plt.show()
    