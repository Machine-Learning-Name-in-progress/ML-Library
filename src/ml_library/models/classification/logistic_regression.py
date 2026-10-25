import warnings

import numpy as np


class logistic_regression:
    def __init__(
        self, learn_rate: float = 0.1, max_iter: int = 1000, tol: float = 1e-6
    ):

        if not isinstance(learn_rate, (int, float)):
            raise TypeError("'learn_rate' must be a number")
        if learn_rate <= 0:
            raise ValueError("'learn_rate' must be greater than 0")

        if not isinstance(max_iter, int):
            raise TypeError("'max_iter' must be an integer")
        if max_iter <= 0:
            raise ValueError("'max_iter' must be greater than 0")

        if not isinstance(tol, (int, float)):
            raise TypeError("'tol' must be a number")
        if tol <= 0:
            raise ValueError("'tol' must be greater than 0")

        self._beta = None
        self._learn_rate = learn_rate
        self._max_iter = max_iter
        self._tol = tol

    def _add_ones_column(self, x):
        b_0 = np.ones(x.shape[0])

        return np.column_stack((b_0, x))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict_prob(self, x):
        if self._beta is None:
            raise ValueError("Cannot predict prob without training")

        x = np.asarray(x)
        x = self._add_ones_column(x)

        z = x @ self._beta

        return self._sigmoid(z)

    def train(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        if False in ((y == 0) | (y == 1)):
            raise ValueError("Logistic regression only works with two classes 0/1")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "Each point from x needs a result in y: same number of rows"
            )

        x = self._add_ones_column(x)
        N, p = x.shape

        beta = np.zeros(p)

        for _ in range(self._max_iter):
            z = x @ beta
            prob = self._sigmoid(z)
            error = y - prob

            gradient = (np.transpose(x) @ error) / N
            step = self._learn_rate * gradient

            beta = beta + step

            if np.linalg.norm(step) < self._tol:
                break
        else:
            warnings.warn(
                f"Did not converge in max_iter = {self._max_iter} iterations",
                RuntimeWarning,
            )

        self._beta = beta

    def predict(self, x):
        probs = self.predict_prob(x)

        prediction = probs > 0.5

        prediction = prediction.astype(int)

        return prediction
