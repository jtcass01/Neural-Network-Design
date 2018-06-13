
"""
We want to design a perceptron network to output a 1 when either of these two vectors are input to the network:
    [-1, 0].T       [1, 2].T
and to output a -1 when either of the following vectors are input to the network:
    [-1, 1].T       [0, 2].T

    i.   Find and sketch a decision boundary for a network that will solve this problem
    ii.  Find weights and biases that will produce the decision boundary you fond in part 1.  Show all work.
    iii. Draw the network diagram using abreviated notation.
    iv.  For each of the four vectors given above, calculate the net input, n, and the network output, a, for the network you have designed.
         Verify that your network sovles the problem.
    v.   Are there other weights and biases that would solve the problem? If so, would you consider your weights best? Explain. 
"""

import numpy as np
import matplotlib.pyplot as plt

from transfer_functions import hardlims, hardlim
from Perceptron import Perceptron


if __name__ == "__main__":
    print("We want to design a perceptron network to output a 1 when either of these two vectors are input to the network:",
          "\n\t[-1, 0].T       [1, 2].T",
          "\nand to output a -1 when either of the following vectors are input to the network:",
          "\n\t[-1, 1].T       [0, 2].T")

    print("\ni.   Find and sketch a decision boundary for a network that will solve this problem")
    x = np.linspace(-3,3, 100)
    y = x + 1.5

    plt.plot(x, y)
    plt.grid(True)
    plt.show()

    print("\nii.  Find weights and biases that will produce the decision boundary you fond in part 1.  Show all work.")
    W = np.array([1, -1]).reshape((1,2))
    b = 1.5
    print("Weights: ", W, "bias", b, "Weight and bias calculated by finding the orthogonal line to the decision boundary sketched above",
          "\n\tWeight associated to slope, bias associated to y intercept in the equation y=mx+b")

    e3_5_perceptron = Perceptron(W=W, b=b)

    o_1_1 = np.array([-1, 0]).reshape((2, 1))
    o_1_2 = np.array([1, 2]).reshape((2, 1))
    o_m1_1 = np.array([-1, 1]).reshape((2, 1))
    o_m1_2 = np.array([0, 2]).reshape((2, 1))

    print("\niii. Draw the network diagram using abreviated notation.")
    print("Passing on this since this is a programmatic implementation of these problems.  Also it seems rather trivial at this point.")

    print("\niv.  For each of the four vectors given above, calculate the net input, n, and the network output, a, for the network you have designed.",
          "\n\tVerify that your network sovles the problem.")
    print("\t\t[-1, 0].T", e3_5_perceptron.classify(o_1_1))
    print("\t\t[1, 2].T", e3_5_perceptron.classify(o_1_2))
    print("\t\t[-1, 1].T", e3_5_perceptron.classify(o_m1_1))
    print("\t\t[0, 2].T", e3_5_perceptron.classify(o_m1_2))

    print("v.   Are there other weights and biases that would solve the problem? If so, would you consider your weights best? Explain. ")
    print("As described in part ii.  This is the absolute best weight and bias as it was calculated from a median decision boundary equation.")
