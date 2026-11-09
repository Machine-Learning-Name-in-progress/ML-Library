
import numpy as np

def _gini(y):
    """return the Gini impurity of a non-empty set of class labels"""
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()
    return 1 - np.sum(proportions ** 2)

def _majority_choice(y):
    columns, counts = np.unique(y, return_counts=True)
    return columns[np.argmax(counts)]


def _mse(y):
    """return the MSE impurity"""
    _mean = np.mean(y)
    return np.mean((y - _mean)**2)

def _mean_choice(y):
    return np.mean(y)

def _split_score(y_left, y_right, impurity):
    """return the sample-weighted Gini/Regressor impurity of two non-empty children"""
    n_left = len(y_left)
    n_right = len(y_right)

    if n_left == 0 or n_right == 0:
        raise ValueError("A split must leave at least one sample on each side.")

    total = n_left + n_right
    return (n_left * impurity(y_left) + n_right * impurity(y_right)) / total

def _best_split(x, y, impurity, min_sample_leaf):
    """return the feature index and threshold of the best split or None"""
    best_score = float("inf")
    best_split = None

    for j in range(x.shape[1]):
        values = np.unique(x[:, j])

        # only the boundaries between consecutive distinct values are candidates
        for lower, upper in zip(values[:-1], values[1:]):
            threshold = float(lower) / 2 + float(upper) / 2
            left = x[:, j] <= threshold

            if not left.any() or left.all() or (np.size(y[left]) < min_sample_leaf or np.size(y[~left]) < min_sample_leaf):
                continue

            score = _split_score(y[left], y[~left], impurity)

            if score < best_score:
                best_score = score
                best_split = (j, threshold)

    return best_split

def _is_pure(y: np.ndarray) -> bool:
    return bool(np.unique(y).size <= 1)

