"""
Perceptron example from pages 3-3:3-8 -- the apple/orange fruit classifier.

The Perceptron class itself lives in nnd/perceptron.py.
"""

import numpy as np

from nnd import Perceptron
from nnd.transfer_functions import hardlims

if __name__ == "__main__":
    # prototype = [shape, texture, weight] as a column vector
    orange_prototype = np.array([1, -1, -1]).reshape((3, 1))
    apple_prototype = np.array([1, 1, -1]).reshape((3, 1))

    # Weight matrix and bias determined by decision boundary.
    decision_boundary = (orange_prototype != apple_prototype).astype(int).reshape((1, len(orange_prototype)))
    print(decision_boundary)
    bias = 0

    fruit_perceptron = Perceptron(W=decision_boundary, b=bias, transfer_function=hardlims)

    test_prototype = np.array([-1, -1, -1]).reshape((3, 1))
    print(fruit_perceptron.classify(test_prototype))
