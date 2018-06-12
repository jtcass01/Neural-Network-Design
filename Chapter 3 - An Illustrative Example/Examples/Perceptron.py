import numpy as np

from transfer_functions import *

class Perceptron(object):
    """
    Implementation of network described on pages 3-3:3-8

    """
    def __init__(self, W, b, transfer_function):
        self.Weights = W
        self.bias = b
        self.transfer_function = transfer_function
    def classify(self, prototype):
        return self.transfer_function(self.Weights.dot(prototype) + self.bias)


if __name__ == "__main__":
    # prototype = [shape, texture, weight] as a column vector
    orange_prototype = np.array([1, -1, -1]).reshape((3, 1))
    apple_prototype = np.array([1, 1, -1]).reshape((3, 1))

    # Weight matrix and bias determined by decision boundary.
    decision_boundary = (orange_prototype != apple_prototype).astype(np.int).reshape((1, len(orange_prototype)))
    bias = 0

    fruit_perceptron = Perceptron(W=decision_boundary, b=bias, transfer_function=hardlims)

    test_prototype = np.array([-1, -1, -1]).reshape((3, 1))
    print(fruit_perceptron.classify(test_prototype))

