
"""
We have the following two prototype vectors:
    p1 = [-1, 1].T    p2 = [1,1].T

i.    Find and sketch a decision boundary for a perceptron network that will recognize these two vectors.
ii.   Find weights and bias that will produce the decision boundary you found in part i.
iii.  Draw the network diagram using abreviated notation.
iv.   For the vector given below, calculate the net input, n, and the network output, a, for the network you have designed.  Does the network produce a good output? Explain.
      p_test = [0.5, -0.5].T
v.    Design a Hamming network to recognize the two vectors used in part i.
vi.   Calculate the network output for the Hamming network for the input vector given in part iv.  Does the network produce a good output? Explain.
vii.  Design a Hopfield network to recognize the two vectors used in part i.
viii. Calculate the network output for the Hopfield network for the input vector given in part iv.  Does the network produce a good output? Explain.
"""

import numpy as np
import matplotlib.pyplot as plt

from Perceptron import Perceptron
from HammingNetwork import HammingNetwork
from HopfieldNetwork import HopfieldNetwork

if __name__ == "__main__":
    print("We have the following two prototype vectors:",
          "\n\tp1 = [-1, 1].T    p2 = [1,1].T")
    p1 = np.array([-1, 1]).reshape((2, 1))
    p2 = np.array([1, 1]).reshape((2, 1))
    prototypes = np.array([p1.T[0], p2.T[0]])

    x = np.linspace(0, 0, 100)
    y = np.linspace(0, 2, 100)

    print("\ni.    Find and sketch a decision boundary for a perceptron network that will recognize these two vectors.")
    plt.scatter(prototypes[0], prototypes[1])
    plt.plot(x, y, color='black')
    plt.grid(True)
    plt.show()

    print("\nii.   Find weights and bias that will produce the decision boundary you found in part i.")
    Weights = np.array([1, 0])
    bias = 0

    e3_6_perceptron = Perceptron(W = Weights, b = bias)

    print("p1", p1, "classification:", e3_6_perceptron.classify(p1))
    print("p2", p2, "classification:", e3_6_perceptron.classify(p2))

    print("\niv.   For the vector given below, calculate the net input, n, and the network output, a, for the network you have designed.  Does the network produce a good output? Explain.",
          "\n\tp_test = [0.5, -0.5].T")

    p_test = np.array([0.5, -0.5]).reshape((2,1))
    print("p_test Classification = ", e3_6_perceptron.classify(p_test))
    print("This is a good classification because the test array lands on the right side of the purposed decision boundary and gets classified as such.")

    print("\nv.    Design a Hamming network to recognize the two vectors used in part i.")
    e3_6_hamming = HammingNetwork(prototypes=prototypes)
    print("p1", p1, "classification:", e3_6_hamming.classify(p1))
    print("p2", p2, "classification:", e3_6_hamming.classify(p2))

    print("\nvi.   Calculate the network output for the Hamming network for the input vector given in part iv.  Does the network produce a good output? Explain.")
    print("p_test Classification = ", e3_6_hamming.classify(p_test))
    print("This is a correct classification but only by chance as the Hamming Network is only designed to classify prototypes that contain two possible values only.")

    print("\nvii.  Design a Hopfield network to recognize the two vectors used in part i.")
    Weights = np.array([[1.2, 0], [0, 0.2]])
    bias = np.array([0, 0.9])

    e3_6_hopfield = HopfieldNetwork(weights=Weights, bias = bias)
    print("p1", p1, "classification:", e3_6_hopfield.classify(p1))
    print("p2", p2, "classification:", e3_6_hopfield.classify(p2))

    print("\nviii. Calculate the network output for the Hopfield network for the input vector given in part iv.  Does the network produce a good output? Explain.")
    print("p_test Classification = ", e3_6_hopfield.classify(p_test))
    print("Yes, this is the same classification outcome as expected and previously calculated from the other two networks.")
    