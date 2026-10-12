"""Activation functions for neural networks."""

import numpy as np


class ReLU:
    """ReLu Activation Function class."""

    def __init__(self) -> None:
        """Initialize the ReLU activation function."""
        self.input_layer = None

    def forward(self, input_layer: np.ndarray) -> np.ndarray:
        """Forward pass for ReLU activation function.

        Args:
            input_layer: Input layer of the neural network.

        Returns:
            Output layer of the neural network.
        """
        self.input_layer = input_layer
        return np.maximum(0, input_layer)

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass for ReLU activation function.

        Args:
            gradient: Gradient of the loss function with
                respect to the output layer.

        Returns:
            Gradient of the loss function with
                respect to the input layer.
        """
        return gradient * (self.input_layer > 0)


class Sigmoid:
    """Sigmoid Activation Function class."""

    def __init__(self) -> None:
        """Initialize the Sigmoid activation function."""
        self.input_layer = None
        self.output_layer = None

    def forward(self, input_layer: np.ndarray) -> np.ndarray:
        """Forward pass for Sigmoid activation function.

        Args:
            input_layer: Input layer of the neural network.

        Returns:
            Output layer of the neural network.
        """
        self.input_layer = input_layer
        self.output_layer = 1 / (1 + np.exp(-input_layer))

        return self.output_layer

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass for Sigmoid activation function.

        Args:
            gradient: Gradient of the loss function with
                respect to the output layer.

        Returns:
            Gradient of the loss function with
                respect to the input layer.
        """
        return gradient * (self.output_layer * (1 - self.output_layer))
