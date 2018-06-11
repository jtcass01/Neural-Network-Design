import numpy as np

from transfer_functions import *

"""
Consider a single-input neuron with a bias.  We would like the output to be -1 for inputs less than 3 and +1 fo rinputs greater than or equal to 3.
    i.   What kind of transfer function is required?
    ii.  What kind of bias would you suggest? Is your bias in any way related to the input weight? If yes, how?
    iii. Summaraize your network by naming the transfer function and stating the bias and the weight.  Draw a diagram of the network.  Verify the network
    performance using MATLAB.
"""

def calculate_net_input(W=np.array(1), p=np.array(2.0), b=-3):
    return W.dot(p) + b

def neuron(W, p, activation_function = hardlims):
    print("\nExample Neuron: ")
    b = -W*p + p/3 - 1
    print("W: ", W, "p: ", p, "b: ", b)
    net_input = calculate_net_input(W=W, p=p, b=b)
    print("net_input", net_input)
    activation = activation_function(net_input)
    print("activation: ", activation)

    assert (activation == 1 and p >= 3) or (activation == -1 and p < 3)

if __name__ == "__main__":
    print("Consider a single-input neuron with a bias.  We would like the output to be -1 for inputs less than 3 and +1 for inputs greater than or equal to 3.")
    print("i.   What kind of transfer function is required?", "hardlims [Symmetrical Hard Limit]")
    print("ii.  What kind of bias would you suggest? Is your bias in any way related to the input weight? If yes, how?",
          "I'd suggest a bias that is -W*p + p/3 - 1.")
    print("iii. Summaraize your network by naming the transfer function and stating the bias and the weight.  Draw a diagram of the network.  Verify the network performance using MATLAB.")

    p_range = np.array(range(-50,50,1))
    p_range = p_range*0.1

    W_range = np.array(range(-50,50,1))
    W_range = W_range*0.1

    for p in p_range:
        for W in W_range:
            neuron(W=np.array(W), p=np.array(p))

    print("Note: I am going to skip drawing the neuron.")