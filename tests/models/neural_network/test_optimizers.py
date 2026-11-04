"""Tests for optimizers"""

import numpy as np
import pytest

from ml_library.models.neural_network.layers import Dense
from ml_library.models.neural_network.optimizers import SGD


def test_sgd_invalid_parameters() -> None:
    """Test if invalid learning rate or momentum raises ValueError."""
    with pytest.raises(ValueError):
        SGD(lr=0.0)
    with pytest.raises(ValueError):
        SGD(lr=-0.01)
    with pytest.raises(ValueError):
        SGD(lr=0.01, momentum=-0.5)
    with pytest.raises(ValueError):
        SGD(lr=0.01, momentum=1.0)


def test_sgd_step_on_dense_layer() -> None:
    """Test SGD step updating simulated weights and bias on a Dense layer."""
    layer = Dense(n_inputs=2, n_outputs=2)
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer.bias = np.array([[0.5, -0.5]])

    layer.grad_weights = np.array([[0.5, -1.0], [2.0, 0.0]])
    layer.grad_bias = np.array([[1.0, -2.0]])

    optimizer = SGD(lr=0.1, momentum=0.0)
    optimizer.step([layer])

    expected_weights = np.array([[0.95, 2.1], [2.8, 4.0]])
    expected_bias = np.array([[0.4, -0.3]])

    np.testing.assert_allclose(layer.weights, expected_weights, rtol=1e-7)
    np.testing.assert_allclose(layer.bias, expected_bias, rtol=1e-7)


def test_sgd_momentum_update() -> None:
    """Test SGD update with momentum across two consecutive steps."""
    optimizer = SGD(lr=0.1, momentum=0.9)
    w = np.array([10.0])
    grad = np.array([2.0])

    w = optimizer.update(w, grad, key="w")
    np.testing.assert_allclose(w, np.array([9.8]), rtol=1e-7)

    w = optimizer.update(w, grad, key="w")
    np.testing.assert_allclose(w, np.array([9.42]), rtol=1e-7)


def test_sgd_parabola_convergence() -> None:
    """Test SGD convergence on a simple parabola f(w) = w^2."""
    optimizer = SGD(lr=0.1, momentum=0.5)
    w = np.array([5.0, -3.0])

    for _ in range(100):
        grad = 2.0 * w
        w = optimizer.update(w, grad, key="parabola")

    np.testing.assert_allclose(w, np.zeros_like(w), atol=1e-6)