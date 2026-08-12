"""
Hamming network example from pages 3-8:3-12 -- the apple/orange fruit classifier.

The HammingNetwork class itself lives in nnd/hamming.py.
"""

import numpy as np

from nnd import HammingNetwork

if __name__ == "__main__":
    # prototype = [shape, texture, weight] as a column vector
    orange_prototype = np.array([1, -1, -1]).reshape((3, 1))
    apple_prototype = np.array([1, 1, -1]).reshape((3, 1))
    prototypes = np.array([orange_prototype.T[0], apple_prototype.T[0]])

    test_fruit = np.array([-1, -1, -1]).reshape((3, 1))

    hammingFruitClassifier = HammingNetwork(prototypes=prototypes)
    print(hammingFruitClassifier.classify(obj=test_fruit))
