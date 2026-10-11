"""Test module for activation functions in the neural network models."""

import numpy as np
from ml_library.models.neural_network.activations import ReLU, Sigmoid


def test_relu_forward():
    """Test for ReLU activation function forward pass."""
    relu = ReLU()
    x = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
    result = relu.forward(x)
    expected = np.array([0.0, 0.0, 0.0, 2.0, 5.0])

    np.testing.assert_array_equal(result, expected)


def test_relu_backward():
    """Test for ReLU activation function backward pass."""
    relu = ReLU()
    x = np.array([-2.0, 0.0, 4.0])
    relu.forward(x)

    da = np.ones_like(x)
    gradient = relu.backward(da)
    expected_gradient = np.array([0.0, 0.0, 1.0])

    np.testing.assert_array_equal(gradient, expected_gradient)


def test_sigmoid_forward():
    """Test for Sigmoid activation function forward pass."""
    sigmoid = Sigmoid()
    x = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
    result = sigmoid.forward(x)
    expected = np.array(
        [
            0.0474258727,
            0.2689414214,
            0.5,
            0.8807970779778823,
            0.9933071490757153,
        ]
    )

    np.testing.assert_array_almost_equal(result, expected, decimal=4)


def test_sigmoid_backward():
    """Test for Sigmoid activation function backward pass."""
    sigmoid = Sigmoid()
    x = np.array([0.5, 1.0, 2.0, 3.0])
    sigmoid.forward(x)

    da = np.ones_like(x)
    gradient = sigmoid.backward(da)
    expected_gradient = np.array(
        [
            0.2350037,
            0.1966119,
            0.1049936,
            0.0451767,
        ]
    )

    np.testing.assert_array_almost_equal(
        gradient, expected_gradient, decimal=7
    )
