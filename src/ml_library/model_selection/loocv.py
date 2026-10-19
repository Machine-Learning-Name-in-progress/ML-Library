"""Leave-One-Out Cross-Validatn splitter."""

import numpy as np


def loocv_indices(n: int) -> list:
    """Split observation for LOOCV.

    Parameters:
        n (int): Number of observations.

    Returns:
        list: Pairs of train and validation indices.

    """
    if n < 2:
        raise ValueError("The number of observations must be at least 2")

    indices = np.arange(n)
    splits = []

    for i in range(n):
        validation_indices = np.array([indices[i]])
        train_indices = np.delete(indices, i)
        splits.append((train_indices, validation_indices))

    return splits
