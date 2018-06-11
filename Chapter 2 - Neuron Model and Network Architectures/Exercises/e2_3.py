import numpy as np
from transfer_functions import *

"""Given a two-input neuron with the folwing weight matrix and input vector: W = [3, 2] and p = [-5, 7], we would like to have an output of 0.5.
Do you suppose that there is a combination of bias and transfer function that might allow this?
    i.   Is there a transfer function from Table 2.1 that will do the job if the bias is zero.
    ii.  Is there a bias that will do the job if the linear transfer function is used? If yes, what is it?
    iii. Is there a bias that will do the job if a log-sigmoid transfter function is used? If yes, what is it?
    iv.  Is there a bias that will do the job if a symmetrical hard limit transfer function is used?
"""

from e2_1 import calculate_net_input

if __name__ == "__main__":
    print("Given a two-input neuron with the folwing weight matrix and input vector: W = [3, 2] and p = [-5, 7], we would like to have an output of 0.5. Do you suppose that there is a combination of bias and transfer function that might allow this?")

    W = np.array([3,2]).reshape((1,2))
    p = np.array([-5,7]).reshape((2,1))

    print("\ni. Is there a transfer function from Table 2.1 that will do the job if the bias is zero.")
    net_input = calculate_net_input(W=W, p=p, b=0)[0][0]
    print("net_input w/ bias=0", net_input)
    print("No. closest is log sigmoid which value is ", logsig(net_input))

    print("\nii.  Is there a bias that will do the job if the linear transfer function is used? If yes, what is it?")
    net_input = calculate_net_input(W=W, p=p, b=1.5)[0][0]
    print("net_input w/ bias=1.5: ", net_input)
    print("Output w/ linear transfer function: ", purelin(net_input))

    print("\niii. Is there a bias that will do the job if a log-sigmoid transfter function is used? If yes, what is it?")
    net_input = calculate_net_input(W=W, p=p, b=1)[0][0]
    print("net_input w/ bias=1: ", net_input)
    print("Output w/ log-sigmoid transfer function: ", logsig(net_input))

    print("\niv.  Is there a bias that will do the job if a symmetrical hard limit transfer function is used?")
    print("No.")

