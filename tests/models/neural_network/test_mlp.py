import numpy as np
import pytest

from ml_library.models.neural_network.activations import Linear, ReLU, Sigmoid, Tanh
from ml_library.models.neural_network.mlp import MLPRegressor
from ml_library.utils import _random

@pytest.mark.parametrize("activation", [Linear, ReLU, Sigmoid, Tanh])
@pytest.mark.parametrize("column_target", [False, True]) #test both 1d and column shapped targets
def test_mlp_learns_regression(monkeypatch, activation, column_target):
    monkeypatch.setattr(_random, "_rng", np.random.default_rng(42))

    X_train = np.linspace(-1, 1, 21).reshape(-1,1)
    y_train = 2 * X_train[:, 0] + 1
    if column_target:
        y_train = y_train.reshape(-1,1)

    X_test = np.array([[-0.85], [-0.35], [0.25], [0.75]])
    y_test = 2 * X_test[:, 0] + 1

    model = MLPRegressor(activation = activation).fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert predictions.shape == y_test.shape
    assert np.mean((predictions - y_test) ** 2) < 0.02


#reproducibility test
def test_mlp_is_reproducible(monkeypatch):
    X = np.linspace(-1, 1, 67).reshape(-1,1)
    y = 2 * X[:, 0] + 1

    monkeypatch.setattr(_random, "_rng", np.random.default_rng(42))
    ref = MLPRegressor().fit(X, y).predict(X) #reference

    monkeypatch.setattr(_random, "_rng", np.random.default_rng(42))
    second_model = MLPRegressor().fit(X, y)

    for i in range(2, 62):
        monkeypatch.setattr(_random, "_rng", np.random.default_rng(42))
        predictions = MLPRegressor().fit(X, y).predict(X)

        np.testing.assert_allclose(
            predictions,
            ref,
            err_msg=f"predictions differed on run {i}"
        )