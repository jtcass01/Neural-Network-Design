import numpy as np

from transfer_functions import *

class Perceptron(object):
    """
    Implementation of network described on pages 3-3:3-8

    An output of 1 means the input p is orthogonal of the decision boundary and points in the same direction as the weight.
    An oupout of -1 means the input p points in the opposite direction of the weight.

    Author: Jacob Taylor Cassady
    """
    def __init__(self, W, b, transfer_function=hardlims):
        self.Weights = W
        self.bias = b
        self.transfer_function = np.vectorize(transfer_function)
    def classify(self, prototype):
        net_input = self.Weights.dot(prototype) + self.bias
        return self.transfer_function(net_input)


if __name__ == "__main__":
    # prototype = [shape, texture, weight] as a column vector
    orange_prototype = np.array([1, -1, -1]).reshape((3, 1))
    apple_prototype = np.array([1, 1, -1]).reshape((3, 1))

    # Weight matrix and bias determined by decision boundary.
    decision_boundary = (orange_prototype != apple_prototype).astype(np.int).reshape((1, len(orange_prototype)))
    print(decision_boundary)
    bias = 0

    fruit_perceptron = Perceptron(W=decision_boundary, b=bias)

    test_prototype = np.array([-1, -1, -1]).reshape((3, 1))
    print(fruit_perceptron.classify(test_prototype))

