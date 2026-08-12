import numpy as np

from nnd.transfer_functions import hardlim

class Perceptron(object):
    """
    Implementation of network described on pages 3-3:3-8

    An output of 1 means the input p is orthogonal of the decision boundary and points in the same direction as the weight.
    An oupout of -1 means the input p points in the opposite direction of the weight.

    The network can be built two ways:
      - with a known decision boundary, by passing W and b directly (Chapter 3), or
      - with random starting values, by passing number_of_neurons and input_size
        and then calling train() (Chapter 4).

    Author: Jacob Taylor Cassady
    """
    def __init__(self, W=None, b=None, number_of_neurons=None, input_size=None, transfer_function=hardlim):
        if W is None:
            if number_of_neurons is None or input_size is None:
                raise ValueError("Pass either W and b, or number_of_neurons and input_size.")
            W = np.random.rand(number_of_neurons, input_size)
            b = np.random.rand(number_of_neurons, 1)

        self.Weights = W
        self.bias = b
        self.transfer_function = np.vectorize(transfer_function)

    def classify(self, prototype):
        net_input = self.Weights.dot(prototype) + self.bias
        return self.transfer_function(net_input)

    def train(self, prototypes):
        old_weights = self.Weights
        old_bias = self.bias

        while not self.correct(prototypes):
            for prototype in prototypes:
                input_v = prototype[0]
                target = prototype[1]

                classification = self.classify(prototype = input_v)

                self.Weights = self.Weights + (target-classification)*input_v.T
                self.bias = self.bias + (target-classification)

    def correct(self, prototypes):
        for prototype in prototypes:
            input_v = prototype[0]
            target = prototype[1]

            if target != self.classify(input_v):
                return False
        return True
