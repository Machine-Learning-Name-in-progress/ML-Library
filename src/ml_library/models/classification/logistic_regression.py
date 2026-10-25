import numpy as np


class logistic_regression:
    def __init__(
        self, learn_rate: float = 0.1, max_iter: int = 1000, tol: float = 1e-6
    ):

        if not isinstance(learn_rate, (int, float)) or learn_rate <= 0:
            raise TypeError("'learn_rate' must be a float number greater than 0")
        elif not isinstance(max_iter, int) or max_iter <= 0:
            raise TypeError("'max_iter' must be a integer greater than 0")
        elif not isinstance(tol, float) or tol <= 0:
            raise TypeError("'tol' must be a float number greater than 0")

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
        x = np.asarray(x)
        x = self._add_ones_column(x)

        z = x @ self._beta

        return self._sigmoid(z)
