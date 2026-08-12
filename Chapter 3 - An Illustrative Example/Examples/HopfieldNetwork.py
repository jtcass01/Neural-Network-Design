"""
Hopfield network example from pages 3-12:3-14 -- the apple/orange fruit classifier.

The HopfieldNetwork class itself lives in nnd/hopfield.py.
"""

import numpy as np

from nnd import HopfieldNetwork

if __name__ == "__main__":
    test_obj = np.array([-1, -1, -1]).reshape((3, 1))

    test_hopfield = HopfieldNetwork()
    print(test_hopfield.classify(test_obj))
