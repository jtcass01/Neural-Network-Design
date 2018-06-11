
import matplotlib.pyplot as plt
import numpy as np

from transfer_functions import *

"""
Sketch the neuron response (plot activation versus p for -2<p<2) for the follwing cases:
    i.   w = 1,  b = 1,  f = hardlims
    ii.  w = -1, b = 1,  f = hardlims
    iii. w = 2,  b = 3,  f = purelin
    iv.  w = 2,  b = 3,  f = satlins
    v.   w = -2, b = -1, f = poslin
"""

def calculate_net_input(W=2.3, p=2.0, b=-3):
    return W.dot(p) + b


def neuron(W, b, p, f):
    a = list([])

    for p_i in p:
        net_input = calculate_net_input(W=W, b=b, p=p_i)
        a.append(f(net_input))

    plot(p, a)

def plot(p, a):
    plt.plot(p,a)
    plt.show()

if __name__ == "__main__":
    p = np.array(range(-200,200,1))
    p = p/100

    print("i.   w = 1,  b = 1,  f = hardlims")
    W = np.array(1)
    neuron(W=W, b=1, p=p, f=hardlims)

    print("ii.  w = -1, b = 1,  f = hardlims")
    W = np.array(-1)
    neuron(W=W, b=1, p=p, f=hardlims)

    print("iii. w = 2,  b = 3,  f = purelin")
    W = np.array(2)
    neuron(W=W, b=3, p=p, f=purelin)

    print("iv.  w = 2,  b = 3,  f = satlins")
    W = np.array(2)
    neuron(W=W, b=3, p=p, f=satlins)

    print("v.   w = -2, b = -1, f = poslin")
    W = np.array(-2)
    neuron(W=W, b=-1, p=p, f=poslin)

    