import numpy as np

from transfer_functions import *

class Perceptron(object):
    """
    Implementation of network described on pages 3-3:3-8

    An output of 1 means the input p is orthogonal of the decision boundary and points in the same direction as the weight.
    An oupout of -1 means the input p points in the opposite direction of the weight.

    Author: Jacob Taylor Cassady
    """
    def __init__(self, number_of_neurons, input_size, transfer_function=hardlim):
        self.Weights = np.random.rand(number_of_neurons, input_size)
        self.bias = np.random.rand(number_of_neurons,1)
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

if __name__ == "__main__":
    p_1 = np.array([1, -1, -1]).reshape((3,1))
    t_1 = np.array([0]).reshape((1,1))

    p_t_1 = np.array([p_1, t_1])

    p_2 = np.array([1, 1, -1]).reshape((3,1))
    t_2 = np.array([1]).reshape((1,1))

    p_t_2 = np.array([p_2, t_2])

    prototypes = np.array([p_t_1, p_t_2])

    print(prototypes[0][0].shape[0])

    test_perceptron = Perceptron(number_of_neurons=1, input_size=prototypes[0][0].shape[0])
    test_perceptron.train(prototypes=prototypes)
    print("Train complete", test_perceptron.Weights, test_perceptron.bias)


    p_1 = np.array([0, 0]).reshape((2,1))
    t_1 = np.array([0]).reshape((1,1))
    p_t_1 = np.array([p_1, t_1])

    p_2 = np.array([0, 1]).reshape((2,1))
    t_2 = np.array([0]).reshape((1,1))
    p_t_2 = np.array([p_2, t_2])

    p_3 = np.array([1, 0]).reshape((2,1))
    t_3 = np.array([0]).reshape((1,1))
    p_t_3 = np.array([p_3, t_3])

    p_4 = np.array([1, 1]).reshape((2,1))
    t_4 = np.array([1]).reshape((1,1))
    p_t_4 = np.array([p_4, t_4])

    prototypes = np.array([p_t_1, p_t_2, p_t_3, p_t_4])

    test_perceptron = Perceptron(number_of_neurons=1, input_size=prototypes[0][0].shape[0])
    test_perceptron.train(prototypes=prototypes)
    print("Train complete", test_perceptron.Weights, test_perceptron.bias)
