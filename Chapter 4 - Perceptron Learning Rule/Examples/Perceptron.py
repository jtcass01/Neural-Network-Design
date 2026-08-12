"""
Perceptron learning rule example -- training a single-neuron perceptron from
random starting weights, first on a 3-input problem and then on an AND gate.

The Perceptron class itself lives in nnd/perceptron.py.
"""

import numpy as np

from nnd import Perceptron, prototype_pairs

if __name__ == "__main__":
    p_1 = np.array([1, -1, -1]).reshape((3,1))
    t_1 = np.array([0]).reshape((1,1))

    p_2 = np.array([1, 1, -1]).reshape((3,1))
    t_2 = np.array([1]).reshape((1,1))

    prototypes = prototype_pairs((p_1, t_1), (p_2, t_2))

    print(prototypes[0][0].shape[0])

    test_perceptron = Perceptron(number_of_neurons=1, input_size=prototypes[0][0].shape[0])
    test_perceptron.train(prototypes=prototypes)
    print("Train complete", test_perceptron.Weights, test_perceptron.bias)


    p_1 = np.array([0, 0]).reshape((2,1))
    t_1 = np.array([0]).reshape((1,1))

    p_2 = np.array([0, 1]).reshape((2,1))
    t_2 = np.array([0]).reshape((1,1))

    p_3 = np.array([1, 0]).reshape((2,1))
    t_3 = np.array([0]).reshape((1,1))

    p_4 = np.array([1, 1]).reshape((2,1))
    t_4 = np.array([1]).reshape((1,1))

    prototypes = prototype_pairs((p_1, t_1), (p_2, t_2), (p_3, t_3), (p_4, t_4))

    test_perceptron = Perceptron(number_of_neurons=1, input_size=prototypes[0][0].shape[0])
    test_perceptron.train(prototypes=prototypes)
    print("Train complete", test_perceptron.Weights, test_perceptron.bias)
