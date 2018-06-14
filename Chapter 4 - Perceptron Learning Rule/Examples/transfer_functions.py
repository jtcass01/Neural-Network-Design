import numpy as np

# Transfer functions as defined by Neural Network Toolbox for MATLAB 

def hardlim(n):
    """
    Hard Limit
    """
    if n < 0:
        return 0
    else:
        return 1

def hardlims(n):
    """
    Symmetrical Hard Limit
    """
    if n < 0:
        return -1
    else:
        return 1

def purelin(n):
    """
    Linear
    """
    return n

def satlin(n):
    """
    Saturing Linear
    """
    if n < 0:
        return 0
    elif n > 1:
        return 1
    else:
        return n

def satlins(n):
    """
    Symmetric Saturating Linear
    """
    if n < -1:
        return -1
    elif n > 1:
        return 1
    else:
        return n

def logsig(n):
    """
    Log-Sigmoid
    """
    return 1/(1+np.exp(-1*n))

def tansig(n):
    """
    Hyperbolic Tangent Sigmoid
    """
    return (np.exp(n) - np.exp(-1*n))/(np.exp(n) + np.exp(-1*n))

def poslin(n):
    """
    Positive Linear
    """
    if n < 0:
        return 0
    else:
        return n

def compet(n):
    """
    Competitive
    """
    return (n > 0).astype(np.int)