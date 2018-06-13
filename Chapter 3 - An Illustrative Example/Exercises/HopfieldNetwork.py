import numpy as np

from transfer_functions import *

class HopfieldNetwork(object):
    """
    Implementation of network described on pages 3-12:3-14

    Author: Jacob Taylor Cassady
    """
    def __init__(self, weights = np.array([[0.2, 0, 0], [0, 1.2, 0], [0, 0, 0.2]]), transfer_function = satlins, bias=np.array([0.9, 0, -0.9])):
        self.Weights = weights
        self.bias = bias.reshape((weights.shape[0], 1))
        self.transfer_function = np.vectorize(transfer_function, otypes=[np.float])
        self.activations = list([])

    def classify(self, a0):
        self.activations.append(a0)
        a1 = self.transfer_function(self.Weights.dot(a0) + self.bias)
        self.activations.append(a1)
        if np.array_equal(a0, a1):
            return a1
        else:
            while True:
                an = self.transfer_function(self.Weights.dot(a1) + self.bias)
                self.activations.append(an)
                if not np.array_equal(an, a1):
                    a1 = an
                else:
                    return an

if __name__ == "__main__":
    test_obj = np.array([-1, -1, -1]).reshape((3, 1))

    test_hopfield = HopfieldNetwork()
    print(test_hopfield.classify(test_obj))