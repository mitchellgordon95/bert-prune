import numpy as np


def prune(tensor, sparsity):
    """Returns the mask that would be used to prune tensor to the specified sparsity"""
    tensor = np.abs(tensor)
    thresh_ind = int(tensor.size * sparsity)
    threshold = np.partition(tensor.flatten(), thresh_ind)[thresh_ind]
    return tensor > threshold
