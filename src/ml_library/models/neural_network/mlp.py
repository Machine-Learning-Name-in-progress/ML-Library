import numpy as np

from .activations import Tanh
from .layers import Dense
from .losses import MSE

class MLPRegressor:
    """Fit  a neural network with onde hidden layer using batch gradient descent"""
    def __init__(
        self, 
        hidden_units: int = 8,
        learning_rate : float = 0.05,
        epochs: int = 500,
    ) -> None:
        #errors
        if not isinstance(hidden_units, (int, np.integer)) or hidden_units < 1:
            raise ValueError("hidden_units must be a positive integer")
        if not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not np.isinfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")


        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden: Dense | None = None
        self._activation = Tanh()
        self._output: Dense | None = None
    
        