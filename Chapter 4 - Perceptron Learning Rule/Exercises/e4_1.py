
"""
Consider the classification problem defined below:
    { p_1 = [-1, 1].T | t_1 = 1 }
    { p_2 = [0, 0].T | t_2 = 1 }
    { p_3 = [1, -1].T | t_3 = 1 }
    { p_4 = [1, 0].T | t_4 = 0 }
    { p_5 = [0, 1].T | t_1 = 0 }

- Draw a graph of the data points, labeled according to their targets.
Is this problem solvable with the network you defined in part (i)? Why or why not?
"""

import numpy as np
from Perceptron import Perceptron
from utilities import *

if __name__ == "__main__":
    # Create prototype 1
    p_1 = np.array([-1, 1]).reshape((2, 1))
    t_1 = np.array([1]).reshape((1, 1))
    p_t_1 = np.array([p_1, t_1])


    # Create prototype 2
    p_2 = np.array([0, 0]).reshape((2, 1))
    t_2 = np.array([1]).reshape((1, 1))
    p_t_2 = np.array([p_2, t_2])

    # Create prototype 3
    p_3 = np.array([1, -1]).reshape((2, 1))
    t_3 = np.array([1]).reshape((1, 1))
    p_t_3 = np.array([p_3, t_3])

    p1 = np.array([p_1, p_2, p_3]).reshape((3, 2))
    
    # Create prototype 4
    p_4 = np.array([1, 0]).reshape((2, 1))
    t_4 = np.array([0]).reshape((1, 1))
    p_t_4 = np.array([p_4, t_4])


    # Create prototype 5
    p_5 = np.array([0, 1]).reshape((2, 1))
    t_5 = np.array([0]).reshape((1, 1))
    p_t_5 = np.array([p_5, t_5])

    p0 = np.array([p_4, p_5]).reshape((2, 2))

    plot_2d_prototypes(p1=p1, p0=p0)

    prototypes = np.array([p_t_1, p_t_2, p_t_3, p_t_4, p_t_5])

    test_perceptron = Perceptron(number_of_neurons=1, input_size=2)
    test_perceptron.train(prototypes=prototypes)

    classify_prototypes(neural_network=test_perceptron, prototypes=prototypes[:, 0], targets=prototypes[:, 1])

    print("This problem is solvable with a perceptron neural network because it is linearly separable")