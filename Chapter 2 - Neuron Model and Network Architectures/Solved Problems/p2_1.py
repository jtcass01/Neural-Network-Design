import numpy as np

"""
The input to a single-input neuron is 2.0, its weight is 2.3 and its bias is -3.
    i.  What is the net input to the transfer function?
    ii. What is the neuron output?
"""

def calculate_net_input(W=2.3, p=2.0, b=-3):
    return W.dot(p) + b

if __name__ == "__main__":
    # i.
    print("i.", calculate_net_input(w=2.3, p=2.0, b=-3))

    # ii.
    print("The output cannot be determined because the transfer function is not specified.")