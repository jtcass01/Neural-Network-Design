import numpy as np

from transfer_functions import *

class HammingNetwork(object):
    """
    Implementation of network described on pages 3-8:3-12
    """
    def __init__(self, prototypes):
        self.feedForwardLayer = self.FeedFowardLayer(W=prototypes)
        self.recurrentLayer = self.RecurrentLayer()

    def classify(self, obj):
        a1 = self.feedForwardLayer.propagate(obj=obj)
        return self.recurrentLayer.propagate(initial_a=a1)

    class FeedFowardLayer(object):
        def __init__(self, W, transfer_function = purelin):
            self.Weights = W
            self.bias = np.array(self.Weights.shape[1]).repeat(self.Weights.shape[0], axis=0).reshape((self.Weights.shape[0], 1))
            self.transfer_function = np.vectorize(transfer_function, otypes=[np.float])
        def propagate(self, obj):
            return self.transfer_function(self.Weights.dot(obj) + self.bias)

    class RecurrentLayer(object):
        def __init__(self, W = np.array([[1, -1/2], [-1/2, 1]]), transfer_function = poslin):
            self.Weights = W
            self.transfer_function = np.vectorize(transfer_function, otypes=[np.float])
            self.bias = 0

        def propagate(self, initial_a):
            a2 = self.transfer_function(self.Weights.dot(initial_a) + self.bias)

            while True:
                a3 = self.transfer_function(self.Weights.dot(a2) + self.bias)
                if a2.all() != a3.all():
                    a2 = a3
                else:
                    return a2


if __name__ == "__main__":
    # prototype = [shape, texture, weight] as a column vector
    orange_prototype = np.array([1, -1, -1]).reshape((3, 1))
    apple_prototype = np.array([1, 1, -1]).reshape((3, 1))
    prototypes = np.array([orange_prototype.T[0], apple_prototype.T[0]])
    
    test_fruit = np.array([-1, -1, -1]).reshape((3, 1))

    hammingFruitClassifier = HammingNetwork(prototypes=prototypes)
    print(hammingFruitClassifier.classify(obj=test_fruit))




