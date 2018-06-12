import numpy as np

class HopfieldNetwork(object):
    def __init__(self, W = np.array([[0.2, 0, 0], [0, 1.2, 0], [0, 0, 0.2]]), transfer_function = satlins, bias=np.array([0.9, 0, 0.9])):
        self.Weights = W.reshape((3, 3))
        self.bias = bias.reshape((3, 1))
        self.transfer_function = np.vectorize(transfer_function, otypes=[np.float])

    def classify(self, initial_a):
        a2 = self.transfer_function(self.Weights.dot(initial_a) + self.bias)

        while True:
            a3 = self.transfer_function(self.Weights.dot(a2) + self.bias)
            if a2.all() != a3.all():
                a2 = a3
            else:
                return a2

if __name__ == "__main__":
    test_obj = np.array([-1, -1, -1]).reshape((3, 1))

    test_hopfield = HopfieldNetwork()
    print(test_hopfield.classify(test_obj))