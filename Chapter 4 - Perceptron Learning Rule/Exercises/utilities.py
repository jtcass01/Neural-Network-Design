
import matplotlib.pyplot as plt

def plot_2d_prototypes(p1, p0):
    print("p1", p1[:, 0], p1.shape)
    print("p0", p0, p0.shape)

    plt.scatter(p1[:, 0], p1[:, 1], cmap="blue")
    plt.scatter(p0[:, 0], p0[:, 1], cmap="red")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

def display_network(neural_network):
    print("\nDisplaying neural network information: ", "\n\tWeights: " + str(neural_network.Weights), "\n\tbias: " + str(neural_network.bias))

def classify_prototypes(neural_network, prototypes, targets=None):
    display_network(neural_network=neural_network)

    if targets is None:
        for prototype in prototypes:
            prototype = prototype.reshape((prototypes.shape[1], 1))
            print("\n\tprototype: ", prototype, "\n\tclassification: ", neural_network.classify(prototype=prototype))
    else:
        for prototype_index, prototype in enumerate(prototypes):
            print("\n\tprototype: ", prototype, "\n\ttarget: ", targets[prototype_index], "\n\tclassification: ", neural_network.classify(prototype=prototype))
