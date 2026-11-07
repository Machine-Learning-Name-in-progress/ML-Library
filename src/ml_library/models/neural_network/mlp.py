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
        activation: type = Tanh
    ) -> None:
        #errors
        if not isinstance(hidden_units, (int, np.integer)) or hidden_units < 1:
            raise ValueError("hidden_units must be a positive integer")
        if not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")


        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation

        self._activation = None
        self._hidden: Dense | None = None
        self._output: Dense | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        """train on a two dimensional feature array with one  target per row"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non_empty two dim array")
        if y.ndim == 1:
            y = y.reshape(-1,1)
        if y.shape != (X.shape[0], 1):
            raise ValueError("y must contain exactly one target per row of x")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite value")

        hidden = Dense(X.shape[1], self.hidden_units)
        activation = self.activation()
        output = Dense(self.hidden_units, 1)
        loss = MSE()

        for _ in range(self.epochs):
            hidden_values = activation.forward(hidden.forward(X))
            predictions = output.forward(hidden_values)
            loss_value = loss.forward(y, predictions)
            if not np.isfinite(loss_value):
                raise FloatingPointError("training produced a not finite loss")
            
            gradient = output.backward(loss.backward())
            hidden.backward(activation.backward(gradient))

            for layer in (hidden, output):
                layer.weights -= self.learning_rate * layer.grad_weights
                layer.bias -= self.learning_rate * layer.grad_bias

        self._hidden = hidden
        self._activation = activation
        self._output = output
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """return onde prediction per row as a one dimensional array"""
        if self._hidden is None or self._activation is None or self._output is None:
            raise RuntimeError("call fit before predict.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._hidden.weights.shape[0]:
            raise ValueError("X must have the same number of features as during fit")

        hidden_values = self._activation.forward(self._hidden.forward(X))
        return self._output.forward(hidden_values).ravel()
        