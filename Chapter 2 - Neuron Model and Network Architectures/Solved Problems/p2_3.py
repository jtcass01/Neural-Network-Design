import numpy as np
from p2_1 import calculate_net_input
from transfer_functions import *

"""
Given a two-input neuron with the following parameters: b = 1.2, W = [3,2], and p = [-5, 6].T, calculate the neuron output for the follwing functions:
    i.   A symmetrical hard limit function
    ii.  A saturating linear transfer function
    iii. A hyperbolic tangent sigmoid (tansig) transfer function
"""

if __name__ == "__main__":
    print("Given a two-input neuron with the following parameters: b = 1.2, W = [3,2], and p = [-5, 6].T, calculate the neuron output for the follwing functions:")
    print("\ti.   A symmetrical hard limit function")
    print("\tii.  A saturating linear transfer function")
    print("\tiii. A hyperbolic tangent sigmoid (tansig) transfer function")

    b = 1.2
    W = np.array([3,2]).reshape((1,2))
    p = np.array([-5, 6]).reshape((2,1))
    print("b", b)
    print("W", W, W.shape)
    print("p", p, p.shape)

    net_input = calculate_net_input(b=b, W=W, p=p)[0][0]

    print("net_input", net_input)

    # i. A symmetrical hard limit function
    print("i. A symmetrical hard limit function", hardlims(net_input))

    # ii. A saturing linear transfer function
    print("ii. A saturing linear transfer function", satlin(net_input))

    # iii. A hyperbolic tangent sigmoid (tansig) transfer function
    print("iii. A hyperbolic tangent sigmoid (tansig) transfer function", tansig(net_input))