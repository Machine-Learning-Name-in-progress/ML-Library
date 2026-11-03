import numpy as np

def _gini(y):
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    proportions = counts / counts.sum()
    return 1 - np.sum(proportions ** 2)

def _split_gini(y_left, y_right):
    n_left = len(y_left)
    n_right = len(y_right)

    if n_left == 0 or n_right == 0:
        raise ValueError("A split must leave at least " \
                        "one sample on each side.")

    total = n_left + n_right
    return (n_left * _gini(y_left) + n_right * _gini(y_right)) / total

def _best_split(x, y):
    best_score = float("inf")
    best_split = None

    for j in range(x.shape[1]): # j:=column
        values = np.unique(x[:, j])

        for lower, upper in zip(values[:-1], values[1:]):
            threshold = float(lower) / 2 + float(upper) / 2
            left = x[:, j] <= threshold

            if not left.any() or left.all():
                continue

            score = _split_gini(y[left], y[~left])

            if score < best_score:
                best_score = score
                best_split = (j, threshold)

    return best_split


class decision_tree_classifier:
    def __init__(self, max_depth: int | None = None) -> None:
        self._max_depth = max_depth
        self._root = None

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        prediction = classes[np.argmax(counts)]
        self._root = {"prediction": prediction}

        if self._max_depth == 0 or len(classes) == 1:
            return
        
        split = _best_split(x, y)

        if split is None:
            return

        column, threshold = split
        left = x[:, column] <= threshold

        left_classes, left_counts = np.unique(
            y[left], return_counts=True
        )
        right_classes, right_counts = np.unique(
            y[~left], return_counts=True
        )

        self._root = {
            "column": column,
            "threshold": threshold,
            "left": {"prediction": left_classes[np.argmax(left_counts)]},
            "right": {"prediction": right_classes[np.argmax(right_counts)]},
        }
        

    def predict(self, x):
        if self._root is None:
            raise ValueError("Call fit before predict.")

        predictions = []

        for sample in np.asarray(x):
            node = self._root

            while "column" in node:
                column = node["column"]
                threshold = node ["threshold"]

                if sample[column] <= threshold:
                    node = node["left"]
                else:
                    node = node["right"]

            predictions.append(node["prediction"])
        return np.array(predictions)