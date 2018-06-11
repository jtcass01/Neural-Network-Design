
"""
A two-layer neural network is to have four inputs and six outputs.  The range of the outputs is to be continuous between 0 and 1.  What can you tell about the
network architecture? Specfically:
    i.   How many neurons are required in each layer?
    ii.  What are the dimensions of the first-layer and second-layer weight matrices?
    iii. What kinds of transfer functions can be used in each layer?
    iv.  Are biases required in either layer?
"""

if __name__ == "__main__":
    print("A two-layer neural network is to have four inputs and six outputs.  The range of the outputs is to be continuous between 0 and 1.  What can you tell about the network architecture?")

    print("\ni.   How many neurons are required in each layer")
    print("Layer 1: ", "Neurons: ", "Not enough information. [We'll call it S1]")
    print("Layer 2: ", "Neurons: ", 6)

    print("\nii.  What are the dimensions of the first-layer and second-layer weight matrices?")
    print("Layer 1: ", "Rows: ", "S2 [# of neurons in Layer 1]", "Columns: ", 4, "\"(S2x4)\"")
    print("Layer 2: ", "Rows: ", 6, "Columns: ", "S2 [# of neurons in Layer 1]", "\"(6xS2)\"")

    print("\niii. What kinds of transfer functions can be used in each layer?")
    print("Layer 1:", "purelin", "satlin", "satlins", "logsig", "tansig", "poslin")
    print("Layer 2:", "satlin", "logsig")

    print("\niv.  Are biases required in either layer?", "No.")