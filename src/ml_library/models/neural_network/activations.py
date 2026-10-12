import numpy as np

class ReLU:
    """Rectified Linear Unit activation"""

    def __init__(self):
        self.input = None

    def forward(self, y: np.ndarray) -> np.ndarray:
        """Apply ReLU and store the input for the backward pass"""
        self.input = y
        return np.maximum(0, y)

    def backward(self, da: np.ndarray) -> np.ndarray:
        """Pass gradients where the stored input is positive; otherwise zero"""
        return da * (self.input > 0)
