"""
Shared network implementations for the Neural Network Design exercises.

Every chapter script imports from here, so there is exactly one copy of each
network and each transfer function in the repository.
"""

from nnd.hamming import HammingNetwork
from nnd.hopfield import HopfieldNetwork
from nnd.perceptron import Perceptron
from nnd.plotting import classify_prototypes, display_network, plot_2d_prototypes
from nnd.prototypes import prototype_pairs

__all__ = [
    "HammingNetwork",
    "HopfieldNetwork",
    "Perceptron",
    "classify_prototypes",
    "display_network",
    "plot_2d_prototypes",
    "prototype_pairs",
]
