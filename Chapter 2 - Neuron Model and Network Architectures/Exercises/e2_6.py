"""
Consider the following NN:

p -> Sat. Linear Layer (2 nodes) - > Linear Layer
Where:
    - W1[1, 1] = 2, B1[1] = 2
    - W1[2, 1] = 1, B1[2] = -1
    - W2 = [1, -1], B2[1] = 0
"""

import numpy as np
import matplotlib.pyplot as plt

from transfer_functions import *

def calculate_net_input(W=np.array(1), p=np.array(2.0), b=-3):
    return W.dot(p) + b

def layer(p, W, b, f):
    n = calculate_net_input(W=W, p=p, b=b)
    f = np.vectorize(f, otypes=[np.float])
    return f(n)

def vectorized_implementation(p, W1=[2,1], b1=[2,1], W2=[1,-1], b2=0):
    input_size = p.shape[0]

    W1 = np.array(W1).reshape((2, 1))
    W1 = W1.repeat(p.shape[0], axis=1)

    b1 = np.array(b1).reshape((2,1))

    a1 = layer(p=p, W=W1, b=b1, f=satlin)   

    W2 = np.array(W2).reshape((2, 1))
    W2 = W2.repeat(a1.shape[0], axis=1)

    b2 = np.array(b2).reshape((1,1))
    a2 = layer(p=a1, W=W2, b=b2, f=purelin)

    return a2

if __name__ == "__main__":
    p = (np.array(range(-3000, 3000, 1)) / 1000).reshape(6000, 1)
    W1=np.array([2,1])
    b1=np.array([2,-1])
    W2=np.array([1,-1])
    b2=0

    a2 = vectorized_implementation(p=p, W1=W1, b1=b1, W2=W2, b2=b2)
    print(a2)

    # Non vectorized implementation for plotting
    n1_1 = list([])
    a1_1 = list([])
    for p_i in p:
        n = calculate_net_input(p=np.array(p_i), W=np.array(W1[0]), b=b1[0])[0]
        a = satlin(n)
        n1_1.append(n)
        a1_1.append(a)
    n1_1 = np.array(n1_1)
    a1_1 = np.array(a1_1)
    
    plt.plot(p, n1_1)
    plt.xlabel("p")
    plt.ylabel("n1_1")
    plt.show()

    plt.plot(p, a1_1)
    plt.xlabel("p")
    plt.ylabel("a1_1")
    plt.show()

    n1_2 = list([])
    a1_2 = list([])
    for p_i in p:
        n = calculate_net_input(p=np.array(p_i), W=np.array(W1[1]), b=b1[1])[0]
        a = satlin(n)
        n1_2.append(n)
        a1_2.append(a)
    n1_2 = np.array(n1_2)
    a1_2 = np.array(a1_2)
    
    plt.plot(p, n1_2)
    plt.xlabel("p")
    plt.ylabel("n1_2")
    plt.show()

    plt.plot(p, a1_2)
    plt.xlabel("p")
    plt.ylabel("a1_2")
    plt.show()

    n2_1 = list([])
    a2_1 = list([])

    for i in range(len(a1_1)):
        n = calculate_net_input(p=np.array([a1_1[i], a1_2[i]]).reshape((2, 1)), W=W2, b=b2)[0]
        a = purelin(n)
        n2_1.append(n)
        a2_1.append(a)
    n2_1 = np.array(n2_1)
    a2_1 = np.array(a2_1)

    plt.plot(p, a2_1)
    plt.xlabel("p")
    plt.ylabel("a2_1")
    plt.show()

