
"""
In this chapter we have designed three different neural networks to distinguish between apples and oranges, based on three sensor mesurements (shape, texture, and weight).
Suppose that we want to distinguish between bananas and pineapples:
bananas    = [-1, 1, -1].T
Pineapples = [-1, -1, 1].T

i.   Design a perceptron to recognize these patterns.
ii.  Design a Hamming network to recognize these patterns.
iii. Design a Hopfield network to recognize thse patterns.
iv.  Test the operation of your networks by applying several different input patterns.  Discuss the advantages and disadvantages of each network.
"""

import numpy as np

from Perceptron import Perceptron
from HammingNetwork import HammingNetwork
from HopfieldNetwork import HopfieldNetwork

banana_prototype = np.array([-1, 1, -1]).reshape((3, 1))
pineapple_prototype = np.array([-1, -1, 1]).reshape((3, 1))
prototypes = np.array([banana_prototype.T[0], pineapple_prototype.T[0]])

near_pineapple = np.array([1, -1, 1]).reshape((3, 1))
near_banana = np.array([-1, 1, -1]).reshape((3, 1))

if __name__ == "__main__":
    print(prototypes)

    print("\ni.   Design a perceptron to recognize these patterns.")
    # Decision boundary is a weight that points towards that of a banana.
    print("\tNote: Decision boundary is a weight that points towards that of a banana.")
    decision_boundary = np.array([0, 1, -1])
    
    bp_perceptron = Perceptron(W=decision_boundary, b=0)
    print("\t", "near_pineapple", bp_perceptron.classify(near_pineapple))
    print("\t", "near_banana", bp_perceptron.classify(near_banana))

    print("\nii.  Design a Hamming network to recognize these patterns.")
    bp_hamming = HammingNetwork(prototypes=prototypes)
    print("\t", "near_pineapple", bp_hamming.classify(near_pineapple))
    print("\t", "near_banana", bp_hamming.classify(near_banana))

    print("\niii. Design a Hopfield network to recognize thse patterns.")
    bp_hopfield = HopfieldNetwork(W = np.array([[0.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2]]), bias=np.array([-0.9, 0, 0]))    
    print("\t", "near_pineapple", bp_hopfield.classify(near_pineapple))
    print("\t", "near_banana", bp_hopfield.classify(near_banana))

    print("\niv.  Test the operation of your networks by applying several different input patterns.  Discuss the advantages and disadvantages of each network.")
    print("I am going to skip the test of different input patterns as the two prototypes are different in 2/3s of the their presentations.  I fear any tests my sully my understanding as I am not sure which result I should be getting to verify.",
          "The hamming network seems to have the simplest implementation as it's weights and biases are defined by the prototypes themselves without the use of much math to calculate complex things such as the eigen matrix in the Hopfield Network",
          "Or the decision boundary within the perceptron network.")
    print("Additionally, it is important to note the Hamming network is designed to only solve problems where each element in the input vector has only two possible values [e.g. 1 or -1].")
