"""Test module for loss functions on neural networks."""

import numpy as np
from ml_library.models.neural_network.losses import (
    MSE,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
)


def test_mse_forward():
    """Test for MSE loss function forward pass."""
    mse = MSE()

    # prediction data:
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # real data:
    y_real = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # act:
    loss = mse.forward(y_pred, y_real)

    # expected:
    expected_loss = 11.0

    # assert:
    np.testing.assert_array_almost_equal(loss, expected_loss, decimal=7)


def test_mse_backward():
    """Test for MSE loss function backward pass."""
    mse = MSE()

    # prediction data:
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # real data:
    y_real = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # act:
    mse.forward(y_pred, y_real)
    gradient = mse.backward()

    # expected:
    expected_gradient = np.array([0.4, 0.8, 1.2, 1.6, 2.0])

    # assert:
    np.testing.assert_array_almost_equal(
        gradient, expected_gradient, decimal=7
    )


def test_categorical_cross_entropy_forward():
    """Test for categorical cross entropy loss function forward pass."""
    cat_cross_entropy = CategoricalCrossEntropy()

    # prediction data:
    y_pred = np.array([0.1, 0.4, 0.5])

    # real data:
    y_real = np.array([0, 1, 0])

    # act:
    loss = cat_cross_entropy.forward(y_pred, y_real)

    # expected:
    expected_loss = 0.9162907

    # assert:
    np.testing.assert_array_almost_equal(loss, expected_loss, decimal=7)


def test_categorical_cross_entropy_backward():
    """Test for categorical cross entropy loss function backward pass."""
    cat_cross_entropy = CategoricalCrossEntropy()

    # prediction data:
    y_pred = np.array([0.3, 0.5, 0.2])

    # real data:
    y_real = np.array([0, 1, 0])

    # act:
    cat_cross_entropy.forward(y_pred, y_real)
    gradient = cat_cross_entropy.backward()

    # expected:
    expected_gradient = np.array([0.0, -2.0, 0.0])

    # assert:
    np.testing.assert_array_almost_equal(
        gradient, expected_gradient, decimal=7
    )


def test_binary_cross_entropy_forward():
    """Test for binary cross entropy loss function forward pass."""
    bin_cross_entropy = BinaryCrossEntropy()

    # prediction data:
    y_pred = np.array([0.1, 0.7, 0.5, 0.2])

    # real data:
    y_real = np.array([0, 1, 0, 1])

    # act:
    loss = bin_cross_entropy.forward(y_pred, y_real)

    # expected:
    expected_loss = 0.6911551

    # assert:
    np.testing.assert_array_almost_equal(loss, expected_loss, decimal=7)


def test_binary_cross_entropy_backward():
    """Test for binary cross entropy loss function backward pass."""
    bin_cross_entropy = BinaryCrossEntropy()

    # prediction data:
    y_pred = np.array([0.9, 0.7, 0.4, 0.3])

    # real data:
    y_real = np.array([1.0, 0.0, 1.0, 1.0])

    # act:
    bin_cross_entropy.forward(y_pred, y_real)
    gradient = bin_cross_entropy.backward()

    # expected:
    expected_gradient = np.array(
        [-0.2777778, 0.8333333, -0.6250000, -0.8333333]
    )

    # assert:
    np.testing.assert_array_almost_equal(
        gradient, expected_gradient, decimal=7
    )
