
"""
A single-layer neural network is to have six inputs and two outputs.  The outputs are to be limited to and continuous over the range 0 to 1.  What can you tell about the network architecture?
Specifically:
    i.   How many neurons are required?
    ii.  What are the dimensions fo the weight matrix?
    iii. What kind of transfer functions could be used?
    iv.  Is a bias required?
"""

def neurons_required(output_size):
    return output_size

def dim_of_weight_matrix(input_size, output_size):
    return (neurons_required(output_size), input_size)

if __name__ == "__main__":
    # i.   How many neurons are required?
    print("i. How many neurons are required?", neurons_required(output_size=2))

    # ii.  What are the dimensions fo the weight matrix?
    print("ii.  What are the dimensions fo the weight matrix?", dim_of_weight_matrix(input_size = 6, output_size=2))

    # iii. What kind of transfer functions could be used?
    print("iii. What kind of transfer functions could be used?", "logsig would be the most appropriate")

    # iv.  Is a bias required?
    print("iv.  Is a bias required?", "Not enough info.")