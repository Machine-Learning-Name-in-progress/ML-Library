import numpy as np

from ml_library.utils._random import get_rng

class Dense:
    """Fully conected neural network layer."""

    def __init__(self, n_inputs: int, n_outputs: int) -> None:
        """Initialize weights, biases, and gradient storage."""
        if n_inputs <= 0 or n_outputs <= 0:
            raise ValueError("Layes dimensions must be positive.")

        # Use the library shared rng.
        rng = get_rng()

        # Scale the initial weights according to the input size.
        self.weights = rng.normal(
            loc = 0.0,
            scale = np.sqrt(1.0 / n_inputs), #detais behind this choice are in documentation
            size = (n_inputs, n_outputs),
        )

        self.bias = np.zeros((1, n_outputs))

        # Store values needed during backpropagation.
        self.input_layer = None
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, input_layer: np.ndarray) -> np.ndarray:
        """COmpute the output of the dense layer

        Args:
            input_layer: Input array of shape (batch_size, n_inputs).
        
        Returns:
            Output array of shape (batch_size, n_outputs).
        """

        # Store the input for use during backpropagation.
        self.input_layer = input_layer

        return input_layer @ self.weights + self.bias

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        if self.input_layer is None:
            raise RuntimeError("Call forward before backward.")
        # Compute gradient for the trainable parameters.
        self.grad_weights = self.input_layer.T @ gradient
        self.grad_bias = np.sum(gradient, axis=0, keepdims=True)

        # Pass the gradient to the previous layer.
        return gradient @ self.weights.T 
