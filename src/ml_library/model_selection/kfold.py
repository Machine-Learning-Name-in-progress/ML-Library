"""Cross-validation splitters."""

import numpy as np


def kfold_indices(n: int, k: int) -> list:
    """Split observation indices into k folds.

    Parameters:
        n (int): Number of observations.
        k (int): Number of folds. Satisfying 2 <= k <= n.

    Returns:
        list: Pairs of training and validation index arrays.
    """
    if k < 2:
        raise ValueError("The number of folds must be at least 2")

    if k > n:
        raise ValueError(
            "The number of folds cannot exceed the number of observations."
        )

    indices = np.arange(n)

    folds = np.array_split(indices, k)

    splits = []

    for i in range(k):
        validation_indices = folds[i]

        train_indices = np.concatenate([folds[j] for j in range(k) if j != i])

        splits.append((train_indices, validation_indices))

    return splits
