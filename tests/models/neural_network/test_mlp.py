"""Test module for Multi-Layer Perceptron (MLP) models."""

import numpy as np
import pytest

from ml_library.models.neural_network.activations import (
    LeakyReLU,
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
)
from ml_library.models.neural_network.mlp import MLPClassifier, MLPRegressor
from ml_library.models.neural_network.optimizers import SGD, Adam, AdamW
from ml_library.utils import _random


@pytest.mark.parametrize("activation", [Linear, ReLU, Sigmoid, Tanh])
@pytest.mark.parametrize("column_target", [False, True])
def test_mlp_learns_regression(monkeypatch, activation, column_target):
    """Test that MLPRegressor learns a simple linear function."""
    monkeypatch.setattr(_random, "_rng", np.random.default_rng(42))

    X_train = np.linspace(-1, 1, 21).reshape(-1, 1)
    y_train = 2 * X_train[:, 0] + 1
    if column_target:
        y_train = y_train.reshape(-1, 1)

    X_test = np.array([[-0.85], [-0.35], [0.25], [0.75]])
    y_test = 2 * X_test[:, 0] + 1

    model = MLPRegressor(activation=activation).fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert predictions.shape == y_test.shape
    assert np.mean((predictions - y_test) ** 2) < 0.02


def test_mlp_is_reproducible(monkeypatch):
    """Verify that repeated fits with the same seed match."""
    X = np.linspace(-1, 1, 67).reshape(-1, 1)
    y = 2 * X[:, 0] + 1

    monkeypatch.setattr(_random, "_rng", np.random.default_rng(42))
    ref = MLPRegressor().fit(X, y).predict(X)

    for i in range(2, 62):
        monkeypatch.setattr(_random, "_rng", np.random.default_rng(42))
        predictions = MLPRegressor().fit(X, y).predict(X)

        np.testing.assert_allclose(
            predictions,
            ref,
            err_msg=f"predictions differed on run {i}",
        )


@pytest.mark.parametrize("activation", [ReLU, LeakyReLU, Tanh])
@pytest.mark.parametrize("optimizer", [SGD, Adam, AdamW])
@pytest.mark.parametrize("one_hot_target", [False, True])
def test_mlp_classifier_multiclass(
    monkeypatch, activation, optimizer, one_hot_target
):
    """Test that MLPClassifier learns a 3-class classification task."""
    monkeypatch.setattr(_random, "_rng", np.random.default_rng(42))

    X = np.array(
        [
            [-2.0, -2.0],
            [-1.8, -2.2],
            [-2.2, -1.9],
            [0.0, 2.0],
            [0.2, 2.1],
            [-0.2, 1.9],
            [2.0, -2.0],
            [2.1, -1.8],
            [1.9, -2.2],
        ]
    )
    y_labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y = np.eye(3)[y_labels] if one_hot_target else y_labels

    lr = 0.1 if optimizer is SGD else 0.05
    clf = MLPClassifier(
        hidden_units=8,
        lr=lr,
        epochs=150,
        activation=activation,
        optimizer=optimizer,
    ).fit(X, y)

    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (9, 3)
    np.testing.assert_array_almost_equal(
        np.sum(probs, axis=1), np.ones(9), decimal=7
    )
    np.testing.assert_array_equal(preds, y_labels)
    assert clf.loss_history[-1] < clf.loss_history[0]
