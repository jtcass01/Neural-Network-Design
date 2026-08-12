import numpy as np

def prototype_pairs(*pairs):
    """
    Pack (input, target) pairs into an (n, 2) array of arrays.

    Each input and target keeps its own shape, so the result has to be an
    object array -- NumPy raises on a ragged np.array([...]) call. Indexing
    works the way the training loop expects: prototypes[:, 0] is every input,
    prototypes[:, 1] is every target, and prototypes[i] is one pair.
    """
    packed = np.empty((len(pairs), 2), dtype=object)

    for index, (input_v, target) in enumerate(pairs):
        packed[index, 0] = input_v
        packed[index, 1] = target

    return packed
