
"""
Consider the following perceptron network.
W = [[1, 1], [-1, 1]]   b = [-2, 0].T   f = hardlims

i.   How many different classes can this network classify?
ii.  Draw a diagram illustrating the regions corresponding to each class.  Label each region with the corresponding network output.
iii. Calculate the network output for the following input.
        p = [1, -1].T
iv.  Plot the input from part iii. in your diagram from part ii, verify that it falls in the correctly labeled region.
"""

import numpy as np

from transfer_functions import hardlims
from Perceptron import Perceptron

if __name__ == "__main__":
    print("Consider the following perceptron network.",
          "\n\tW = [[1, 1], [-1, 1]]   b = [-2, 0].T   f = hardlims")
    
    weights = np.array([[1, 1], [-1, 1]])
    bias = np.array([-2, 0]).reshape((2, 1))

    e3_4_perceptron = Perceptron(W = weights, b = bias, transfer_function = hardlims)
    
    print("\ni.   How many different classes can this network classify?")
    print("4 networks since the possible outputs are [[-1, -1]], [[-1, 1]], [[1, -1]], [[1, 1]]")

    print("\nii.  Draw a diagram illustrating the regions corresponding to each class.  Label each region with the corresponding network output.")
    print("TODO!!!! I need to figure out how to determine these classification edges...")

    print("\niii. Calculate the network output for the following input.", "\n\tp = [1, -1].T")
    test_prototype = np.array([1, -1]).reshape((2, 1))
    print("Output:", e3_4_perceptron.classify(test_prototype))

    print("iv.  Plot the input from part iii. in your diagram from part ii, verify that it falls in the correctly labeled region.")
    print("TODO!!!! I need to figure out how to determine these classification edges...")
