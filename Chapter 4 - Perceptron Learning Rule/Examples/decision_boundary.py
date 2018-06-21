
"""
AND GATE perceptron implementation, find the decision boundary, weights, and bias.

Use hardlim activation
"""

import numpy as np
import matplotlib.pyplot as plt

from transfer_functions import hardlim
from Perceptron import Perceptron

# target = 0
and_00 = np.array([0, 0]).reshape((2, 1))

# target = 0
and_10 = np.array([1, 0]).reshape((2, 1))

# target = 0
and_01 = np.array([0, 1]).reshape((2, 1))

# target = 1
and_11 = np.array([1, 1]).reshape((2, 1))

and_prototypes = np.array([and_00.T[0], and_10.T[0], and_01.T[0], and_11.T[0]])

if __name__ == "__main__":
    print(and_prototypes)

    weights = np.array([1, 1]).reshape((1, 2))
    bias = -1.5

    x = np.linspace(0, 2, 50)
    y = x

    perceptron = Perceptron(W = weights, b = bias, transfer_function = hardlim)

    for prototype in and_prototypes:
        print("prototype", prototype)
        print("Classification: ", perceptron.classify(prototype=prototype.T))

    plt.scatter(and_prototypes[:, 0], and_prototypes[:, 1])
    plt.plot(x,y, color='black')
    plt.grid(True)
    plt.show()
