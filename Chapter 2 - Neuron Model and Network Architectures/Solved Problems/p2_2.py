from transfer_functions import *
from p2_1 import calculate_net_input

"""
What is the output of the neuron of 2_1 if it has the following transfer functions?
    i.   Hard limit
    ii.  Linear
    iii. log-sigmoid
"""

if __name__ == "__main__":
    # i. Hard Limit
    print("i. Hard Limit ", hardlim(calculate_net_input()))
    # ii. Linear
    print("ii. Linear ", purelin(calculate_net_input()))
    # iii. Log-sigmoid
    print("iii. Log-sigmoid ", logsig(calculate_net_input()))