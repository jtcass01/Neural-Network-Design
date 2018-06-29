
"""
Consider the classification problem defined below:
    { p_1 = [-1, 1].T  |  t_1 = 1 }
    { p_2 = [-1, -1].T  |  t_2 = 1 }
    { p_3 = [0, 0].T  |  t_3 = 0 }
    { p_4 = [1, 0].T  |  t_4 = 0 }

(i.)   Design a single-neuron perceptron to solve this problem.  Design the network graphically by choosing weight vectors that are orthogonal to the decision boundaries.
(ii.)  Test your solution with all four input vectors
(iii.) Classify the following input vectors with your solution:
    { p_5 = [-2, 0].T  |  t_5 = ? }
    { p_6 = [1, 1].T  |  t_6 = ? }
    { p_7 = [0, 1].T  |  t_7 = ? }
    { p_8 = [-1, -2].T  |  t_8 = ? }
(iv.)  Which of the vectors in part (iii) will always be classified the same way, regardless of the solution values for W and b?  Which may vary depending on the solution?
       Why?
"""

import numpy as np
from utilities import *
from Perceptron import Perceptron

if __name__ == "__main__":
    print("Consider the classification problem defined below:",
          "\n\t{ p_1 = [-1, 1].T  |  t_1 = 1 }",
          "\n\t{ p_2 = [-1, -1].T  |  t_2 = 1 }",
          "\n\t{ p_3 = [0, 0].T  |  t_3 = 0 }",
          "\n\t{ p_4 = [1, 0].T  |  t_4 = 0 }",
          )
    # Prototype 1
    p_1 = np.array([-1, 1]).reshape((2, 1))
    t_1 = np.array([1]).reshape((1, 1))
    p_t_1 = np.array([p_1, t_1])

    # Prototype 2
    p_2 = np.array([-1, -1]).reshape((2, 1))
    t_2 = np.array([1]).reshape((1, 1))
    p_t_2 = np.array([p_2, t_2])

    # Prototype 3
    p_3 = np.array([0, 0]).reshape((2, 1))
    t_3 = np.array([0]).reshape((1, 1))
    p_t_3 = np.array([p_3, t_3])

    # Prototype 4
    p_4 = np.array([1, 0]).reshape((2, 1))
    t_4 = np.array([0]).reshape((1, 1))
    p_t_4 = np.array([p_4, t_4])

    p0 = np.array([p_3, p_4]).reshape((2, 2))
    p1 = np.array([p_1, p_2]).reshape((2, 2))

    plot_2d_prototypes(p0=p0, p1=p1)

    prototypes = np.array([p_t_1, p_t_2, p_t_3, p_t_4])

    print("\n(i.)   Design a single-neuron perceptron to solve this problem.  Design the network graphically by choosing weight vectors that are orthogonal to the decision boundaries.")
    e4_2_perceptron = Perceptron(number_of_neurons=1, input_size=2)
    e4_2_perceptron.train(prototypes=prototypes)
    display_network(neural_network = e4_2_perceptron)

    print("\n(ii.)  Test your solution with all four input vectors")
    classify_prototypes(neural_network=e4_2_perceptron, prototypes=prototypes[:, 0], targets=prototypes[:, 1])

    print("\n(iii.) Classify the following input vectors with your solution:",
          "\n\t{ p_5 = [-2, 0].T  |  t_5 = ? }",
          "\n\t{ p_6 = [1, 1].T  |  t_6 = ? }",
          "\n\t{ p_7 = [0, 1].T  |  t_7 = ? }",
          "\n\t{ p_8 = [-1, -2].T  |  t_8 = ? }",
          )
    p_5 = np.array([-2, 0]).reshape((2, 1))
    p_6 = np.array([1, 1]).reshape((2, 1))
    p_7 = np.array([0, 1]).reshape((2, 1))
    p_8 = np.array([-1, -2]).reshape((2, 1))

    prototypes_2 = np.array([p_5, p_6, p_7, p_8]).reshape((4, 2))
    classify_prototypes(neural_network=e4_2_perceptron, prototypes=prototypes)

    print("\n(iv.)  Which of the vectors in part (iii) will always be classified the same way, regardless of the solution values for W and b?  Which may vary depending on the solution?",
          "\n\tWhy?")

    plt.scatter(prototypes_2[:, 0], prototypes_2[:, 1], cmap="yellow")
    plot_2d_prototypes(p0=p0, p1=p1)

    print("\tAs you can see from the previous plot, p_5 and p_6 will always be classified the same since they are so far away from any possible decision boundaries.  Pivoting the slope of the decision boundary will never cross these points while still being a successful solution.")

