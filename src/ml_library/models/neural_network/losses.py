"""Losses classes for neural networks."""

import numpy as np


class MSE:
    """Mean Squared Error loss function."""

    def __init__(self) -> None:
        """Initialize the MSE loss function."""
        self.y_real = None
        self.y_pred = None

    def forward(self, y_real: np.ndarray, y_pred: np.ndarray) -> float:
        """Forward pass for MSE loss function.

        Args:
            y_real: Real values.
            y_pred: Predicted values.

        Returns:
            MSE loss value.
        """
        self.y_real = y_real
        self.y_pred = y_pred
        return float(np.mean((self.y_real - self.y_pred) ** 2))

    def backward(self) -> np.ndarray:
        """Backward pass for MSE loss function.

        Returns:
            Gradient of the loss function with respect to y_pred.
        """
        return 2 * (self.y_pred - self.y_real) / self.y_pred.size


class BinaryCrossEntropy:
    """Binary Cross Entropy loss class."""

    def __init__(self, eps: float = 1e-15) -> None:
        """Initialize the Binary Cross Entropy loss function.

        Args:
            eps: Small value to prevent log(0).
        """
        self.eps = eps
        self.y_real = None
        self.y_pred = None

    def forward(self, y_real: np.ndarray, y_pred: np.ndarray) -> float:
        """Forward pass for Binary Cross Entropy loss function.

        Args:
            y_real: Real values.
            y_pred: Predicted values.

        Returns:
            Binary Cross Entropy loss value.
        """
        self.y_real = y_real
        self.y_pred = y_pred
        y_pred_clipped = np.clip(y_pred, self.eps, 1.0 - self.eps)
        loss = -(
            y_real * np.log(y_pred_clipped)
            + (1.0 - y_real) * np.log(1.0 - y_pred_clipped)
        )
        return float(np.mean(loss))

    def backward(self) -> np.ndarray:
        """Backward pass for Binary Cross Entropy loss function.

        Returns:
            Gradient of the loss function with respect to y_pred.
        """
        y_pred_clipped = np.clip(self.y_pred, self.eps, 1.0)
        gradient = (1.0 - self.y_real) / (
            1.0 - y_pred_clipped
        ) - self.y_real / y_pred_clipped
        return gradient / self.y_pred.size


class CategoricalCrossEntropy:
    """Categorical Cross Entropy loss class."""

    def __init__(self, eps: float = 1e-15) -> None:
        """Initialize Categorical Cross Entropy loss function."""
        self.eps = eps
        self.y_real = None
        self.y_pred = None

    def forward(self, y_real: np.ndarray, y_pred: np.ndarray) -> float:
        """Forward pass for Categorical Cross Entropy loss function.

        Args:
            y_real: Real values (one-hot encoding).
            y_pred: Predicted probabilities for each class.

        Returns:
            Categorical Cross Entropy loss value.
        """
        self.y_real = y_real
        self.y_pred = y_pred
        y_pred_clipped = np.clip(y_pred, self.eps, 1.0 - self.eps)

        # return loss
        return float(-np.sum(y_real * np.log(y_pred_clipped)))

    def backward(self) -> np.ndarray:
        """Backward pass for Categorical Cross Entropy loss function.

        Returns:
            Gradient of the loss function with respect to y_pred.
        """
        y_pred_clipped = np.clip(self.y_pred, self.eps, 1.0 - self.eps)
        gradient = -self.y_real / y_pred_clipped
        return gradient
