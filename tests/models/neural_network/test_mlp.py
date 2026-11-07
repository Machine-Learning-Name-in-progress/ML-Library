import numpy as np
import pytest

from ml_library.models.neural_network.mlp import MLPRegressor
from ml_library.utils import _random

@pytest.mark.parametrize("column_target", [False, True]) #test both 1d and column shapped targets
def test_mlp_learns_regression(monkeypatch, column_target):
    monkeypatch.setattr(_random, "_rng", np.random.default_rng(42))

    X_train = np.linspace(-1, 1, 21).reshape(-1,1)
    y_train = 2 * X_train[:, 0] + 1
    if column_target:
        y_train = y_train.reshape(-1,1)

    X_test = np.array([[-0.85], [-0.35], [0.25], [0.75]])
    y_test = 2 * X_test[:, 0] + 1

    model = MLPRegressor().fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert predictions.shape == y_test.shape
    assert np.mean((predictions - y_test) ** 2) < 0.02

    