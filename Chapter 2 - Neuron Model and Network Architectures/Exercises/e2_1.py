import numpy as np

"""
A single input enuron has a weight of 1.3 and a bias of 3.0.  What possible transfer functions from table 2.1 could this enuron have, if its output is given below.  In each case give the input
that would produce these outputs.
    i.   1.6
    ii.  1.0
    iii. 0.9963
    iv.  -1.0

"""

def calculate_net_input(W=2.3, p=2.0, b=-3):
    return W.dot(p) + b

def back_calculate(n, W, b):
    return (n - b)/W

if __name__ == "__main__":
    print("A single input enuron has a weight of 1.3 and a bias of 3.0.  What possible transfer functions from table 2.1 could this neuron have,\
            if its output is given below.  In each case give the input that would produce these outputs.")

    print("Note: I am skipping the calculation of the input.")

    print("i. 1.6", "Transfer functions: (a.) poslin, (b.) purelin")
    print("ii. 1.0", "Transfer functions: (a.) hardlim, (b.) hardlims, (c.) purelin, (d.) satlin, (e.) satlins, (f.)poslin")
    print("iii. 0.9963", "Transfer functions: (a.) purelin, ")