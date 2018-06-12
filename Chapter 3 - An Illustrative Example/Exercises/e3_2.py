
"""
Consider the following prototype patterns:
    p_1 = [1, 0.5].T
    p_2 = [2, 1].T

i.   Find and sketch the decision boundary for a perceptron network that will recognize these two vectors.
ii.  Find weights and bias which will produce the decision boundary you found in part i, and sketch the network diagram.
iii. Calculate the network output for the following input.  Is the network response (decision) reasonable? Explain.
     p_test = [1, 0].T
iv.  Design a Hamming network to recognize the two prototype vectors above.
v.   Calculate the network output for the Hamming network with the input vector given in part iii, showing all steps.
     Does the Hamming network produce the same deceiion as the perceptron? Explain why or why not.  Which network is
     better suited to this problem? Explain.
"""

import numpy as np

from HammingNetwork import HammingNetwork
from Perceptron import Perceptron

p_1 = np.array([1, 0.5]).reshape((2, 1))
p_2 = np.array([2, 1]).reshape((2, 1))


if __name__ == "__main__":
    decision_boundary = (p_1 + p_2)/2
    W = np.array([0.75, 0])
    b = 0.75

    print("\ni.   Find and sketch the decision boundary for a perceptron network that will recognize these two vectors.",
          "\nii.  Find weights and bias which will produce the decision boundary you found in part i, and sketch the network diagram.")
    print("decision_boundary", decision_boundary, "W", W, "b", b)

    print("\niii. Calculate the network output for the following input.  Is the network response (decision) reasonable? Explain.",
          "\n\tp_test = [1, 0].T")
    p_test = np.array([1, 0]).reshape((2, 1))
    test_perceptron = Perceptron(W=W, b=b)
    print("Classification = ", test_perceptron.classify(p_test), "\nYes this is reasonable as p_test is closer to p_1 than p_2")

    print("\niv.  Design a Hamming network to recognize the two prototype vectors above.")
    prototypes = np.array([p_1.T[0], p_2.T[0]])
    test_hamming = HammingNetwork(prototypes=prototypes)
    print("Classification = ", test_hamming.classify(p_test))

    print("\nv.   Calculate the network output for the Hamming network with the input vector given in part iii, showing all steps. Does the Hamming network produce the same deceiion as the perceptron? Explain why or why not.  Which network is better suited to this problem? Explain.")
    print("No.  The Hamming network misclassifies the test input.  This is because the Hamming network is only designed to classify problems where the input vector only has two possible values.")
