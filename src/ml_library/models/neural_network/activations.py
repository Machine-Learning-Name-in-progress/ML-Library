"""Activation functions for neural networks."""

import numpy as np


class ReLU:
    """ReLu Activation Function class."""

    def __init__(self) -> None:
        """Initialize the ReLU activation function."""
        self._input_layer = None

    def forward(self, input_layer: np.ndarray) -> np.ndarray:
        """Forward pass for ReLU activation function.

        Args:
            input_layer: Input layer of the neural network.

        Returns:
            Output layer of the neural network.
        """
        self._input_layer = input_layer
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
        if self._input_layer is None:
            raise RuntimeError("Call forward before backward.")

        return gradient * (self._input_layer > 0)


class Sigmoid:
    """Sigmoid Activation Function class."""

    def __init__(self) -> None:
        """Initialize the Sigmoid activation function."""
        self._input_layer = None
        self._output_layer = None

    def forward(self, input_layer: np.ndarray) -> np.ndarray:
        """Forward pass for Sigmoid activation function.

        Args:
            input_layer: Input layer of the neural network.

        Returns:
            Output layer of the neural network.
        """
        self._input_layer = np.asarray(input_layer, dtype=np.float64)
        # avoid overflow by using a nonpositive exponent
        z = np.exp(-np.abs(self._input_layer))
        # for x >= 0: 1 / (1 + z)
        # for x < 0:  z / (1 + z)
        self._output_layer = np.where(
            self._input_layer >= 0, 1.0 / (1.0 + z), z / (1.0 + z)
        )
        return self._output_layer

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass for Sigmoid activation function.

        Args:
            gradient: Gradient of the loss function with
                respect to the output layer.

        Returns:
            Gradient of the loss function with
                respect to the input layer.
        """
        if self._output_layer is None:
            raise RuntimeError("Call forward before backward.")

        return gradient * (self._output_layer * (1 - self._output_layer))


class Linear:
    """Linear Activation Function class."""

    def forward(self, input_layer: np.ndarray) -> np.ndarray:
        """Forward pass for the linear activation function.

        Args:
            input_layer: Input to the activation function.

        Returns:
            The input unchanged.
        """
        return input_layer

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass for linear activation function.

        Args:
            gradient: Gradient of the loss function with
                respect to the output layer.

        Returns:
            Gradient of the loss function with
                respect to the input layer.
        """
        # the derivative is 1, so the gradient stays the same
        return gradient


class Tanh:
    """Hyperbolic tangent activation function class."""

    def __init__(self) -> None:
        """Initialize the Hyperbolic tangent activation function."""
        self._output_layer = None

    def forward(self, input_layer: np.ndarray) -> np.ndarray:
        """Forward pass for Hyperbolic tangent activation function.

        Args:
            input_layer: Input layer of the neural network.

        Returns:
            Output layer of the neural network.
        """
        # save tanh(x) for the backward pass
        self._output_layer = np.tanh(input_layer)
        return self._output_layer

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass for Hyperbolic tangent activation function.

        Args:
            gradient: Gradient of the loss function with
                respect to the output layer.

        Returns:
            Gradient of the loss function with
                respect to the input layer.
        """
        if self._output_layer is None:
            raise RuntimeError("Call forward before backward.")

        # the derivative is 1 - tanh(x)**2
        return gradient * (1.0 - self._output_layer**2)


class LeakyReLU:
    """Leaky ReLU activation function class."""

    def __init__(self, alpha: float = 0.01) -> None:
        """Initialize the Leaky ReLU activation function.

        Args:
            alpha: Slope for negative inputs.
        """
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative.")
        self.alpha = alpha
        self._input_layer = None

    def forward(self, input_layer: np.ndarray) -> np.ndarray:
        """Forward pass for Leaky ReLU activation function.

        Args:
            input_layer: Input layer of the neural network.

        Returns:
            Output layer of the neural network.
        """
        self._input_layer = np.asarray(input_layer, dtype=np.float64)
        return np.where(
            self._input_layer > 0,
            self._input_layer,
            self.alpha * self._input_layer,
        )

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass for Leaky ReLU activation function.

        Args:
            gradient: Gradient of the loss function with
                respect to the output layer.

        Returns:
            Gradient of the loss function with
                respect to the input layer.
        """
        if self._input_layer is None:
            raise RuntimeError("Call forward before backward.")

        slope = np.where(self._input_layer > 0, 1.0, self.alpha)
        return gradient * slope


class Softmax:
    """Softmax activation function class for multi-class output."""

    def __init__(self, eps: float = 1e-15) -> None:
        """Initialize the Softmax activation function.

        Args:
            eps: Small constant to prevent probability underflow.
        """
        self.eps = eps
        self._output_layer = None

    def forward(self, input_layer: np.ndarray) -> np.ndarray:
        """Forward pass for Softmax activation function.

        Args:
            input_layer: Input logits of the neural network.

        Returns:
            Normalized probability distribution along the last axis.
        """
        x = np.asarray(input_layer, dtype=np.float64)
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        probs = np.clip(probs, self.eps, 1.0)
        self._output_layer = probs / np.sum(
            probs, axis=-1, keepdims=True
        )
        return self._output_layer

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Backward pass for Softmax activation function.

        Args:
            gradient: Gradient of the loss function with
                respect to the output probabilities.

        Returns:
            Gradient of the loss function with
                respect to the input logits.
        """
        if self._output_layer is None:
            raise RuntimeError("Call forward before backward.")

        dot = np.sum(
            gradient * self._output_layer, axis=-1, keepdims=True
        )
        return self._output_layer * (gradient - dot)

