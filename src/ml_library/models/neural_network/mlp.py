"""Multi-Layer Perceptron (MLP) models for regression and classification."""

import numpy as np

from .activations import ReLU, Softmax, Tanh
from .layers import Dense
from .losses import MSE, CategoricalCrossEntropy
from .optimizers import SGD, Adam


class MLPRegressor:
    """Neural network for single-output regression with one hidden layer.

    The hidden-layer activation is configurable, the output layer is
    linear, and training uses full-batch gradient descent to minimize
    mean squared error.

    Args:
        hidden_units: Number of neurons in the hidden layer.
        learning_rate: Step size used by the optimizer.
        epochs: Number of passes over the training data.
        activation: Hidden-layer activation class (e.g., Tanh, ReLU).
    """

    def __init__(
        self,
        hidden_units: int = 8,
        learning_rate: float = 0.05,
        epochs: int = 500,
        activation: type = Tanh,
    ) -> None:
        """Initialize the MLPRegressor."""
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

        self._activation_instance = None
        self._hidden: Dense | None = None
        self._output: Dense | None = None
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        """Train a new regression network on the given samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Regression targets of shape (n_samples,) or (n_samples, 1).

        Returns:
            The fitted model.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non_empty two dim array")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.shape != (X.shape[0], 1):
            raise ValueError("y must contain exactly one target per row of x")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("X and y must contain only finite value")

        hidden = Dense(X.shape[1], self.hidden_units)
        activation = self.activation()
        output = Dense(self.hidden_units, 1)
        loss = MSE()
        optimizer = SGD(lr=self.learning_rate)
        layers = [hidden, output]

        self.loss_history = []

        for _ in range(self.epochs):
            hidden_values = activation.forward(hidden.forward(X))
            predictions = output.forward(hidden_values)
            loss_value = loss.forward(y, predictions)
            if not np.isfinite(loss_value):
                raise FloatingPointError(
                    "training produced a not finite loss"
                )
            self.loss_history.append(float(loss_value))

            gradient = output.backward(loss.backward())
            hidden.backward(activation.backward(gradient))

            optimizer.step(layers)

            for layer in layers:
                if (
                    not np.isfinite(layer.weights).all()
                    or not np.isfinite(layer.bias).all()
                ):
                    raise FloatingPointError(
                        "training produced non-finite parameters"
                    )

        self._hidden = hidden
        self._activation_instance = activation
        self._output = output
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict one value per sample using the fitted network.

        Args:
            X: Feature matrix with the same number of columns used in fit.

        Returns:
            A one-dimensional array of shape (n_samples,).

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If X has the wrong shape or non-finite values.
        """
        if (
            self._hidden is None
            or self._activation_instance is None
            or self._output is None
        ):
            raise RuntimeError("call fit before predict.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._hidden.weights.shape[0]:
            raise ValueError(
                "X must have the same number of features as during fit"
            )

        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        hidden_values = self._activation_instance.forward(
            self._hidden.forward(X)
        )
        return self._output.forward(hidden_values).ravel()


class MLPClassifier:
    """Multi-output neural network for multi-class classification.

    The output layer has one neuron per class followed by Softmax
    activation, trained with Categorical Cross-Entropy loss and a
    configurable optimizer.

    Args:
        hidden_units: Number of neurons in the hidden layer.
        learning_rate: Step size used by the optimizer.
        epochs: Number of passes over the training data.
        activation: Hidden-layer activation class (e.g., ReLU, Tanh).
        optimizer: Optimizer class (e.g., Adam, AdamW, SGD).
        lr: Optional alias for learning_rate.
    """

    def __init__(
        self,
        hidden_units: int = 32,
        learning_rate: float = 0.01,
        epochs: int = 300,
        activation: type = ReLU,
        optimizer: type = Adam,
        lr: float | None = None,
    ) -> None:
        """Initialize the MLPClassifier."""
        if lr is not None:
            learning_rate = lr

        if not isinstance(hidden_units, (int, np.integer)) or hidden_units < 1:
            raise ValueError("hidden_units must be a positive integer.")
        if not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError("epochs must be a positive integer.")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")

        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.lr = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.optimizer = optimizer

        self._hidden: Dense | None = None
        self._activation_instance = None
        self._output: Dense | None = None
        self._output_activation: Softmax | None = None
        self._classes: np.ndarray | None = None
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        """Train the classification network on the given samples.

        Supports both 1-D class labels of shape (n_samples,) and 2-D
        one-hot encoded targets of shape (n_samples, n_classes).

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target labels (1-D) or one-hot matrix (2-D).

        Returns:
            The fitted model.
        """
        X = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional array.")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values.")

        if y_arr.ndim == 1 or (y_arr.ndim == 2 and y_arr.shape[1] == 1):
            y_flat = y_arr.ravel()
            if y_flat.shape[0] != X.shape[0]:
                raise ValueError(
                    "y must have the same number of samples as X."
                )
            self._classes, y_idx = np.unique(y_flat, return_inverse=True)
            n_classes = len(self._classes)
            if n_classes < 2:
                raise ValueError("y must contain at least two classes.")
            y_oh = np.eye(n_classes, dtype=float)[y_idx]
        elif y_arr.ndim == 2 and y_arr.shape[1] >= 2:
            if y_arr.shape[0] != X.shape[0]:
                raise ValueError(
                    "y must have the same number of samples as X."
                )
            y_oh = np.asarray(y_arr, dtype=float)
            if not np.isfinite(y_oh).all():
                raise ValueError("y must contain only finite values.")
            n_classes = y_oh.shape[1]
            self._classes = np.arange(n_classes)
        else:
            raise ValueError("y must be a 1-D label array or 2-D one-hot.")

        hidden = Dense(X.shape[1], self.hidden_units)
        activation = self.activation()
        output = Dense(self.hidden_units, n_classes)
        output_activation = Softmax()
        loss = CategoricalCrossEntropy()
        optimizer = self.optimizer(lr=self.learning_rate)
        layers = [hidden, output]

        self.loss_history = []
        n_samples = X.shape[0]

        for _ in range(self.epochs):
            hidden_values = activation.forward(hidden.forward(X))
            logits = output.forward(hidden_values)
            probabilities = output_activation.forward(logits)
            loss_value = loss.forward(y_oh, probabilities) / n_samples
            if not np.isfinite(loss_value):
                raise FloatingPointError(
                    "training produced a non-finite loss."
                )
            self.loss_history.append(float(loss_value))

            grad_probs = loss.backward() / n_samples
            grad_logits = output_activation.backward(grad_probs)
            grad_hidden = output.backward(grad_logits)
            hidden.backward(activation.backward(grad_hidden))

            optimizer.step(layers)

            for layer in layers:
                if (
                    not np.isfinite(layer.weights).all()
                    or not np.isfinite(layer.bias).all()
                ):
                    raise FloatingPointError(
                        "training produced non-finite parameters."
                    )

        self._hidden = hidden
        self._activation_instance = activation
        self._output = output
        self._output_activation = output_activation
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for each sample.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability matrix of shape (n_samples, n_classes).
        """
        if (
            self._hidden is None
            or self._activation_instance is None
            or self._output is None
            or self._output_activation is None
        ):
            raise RuntimeError("Call fit before predict_proba.")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._hidden.weights.shape[0]:
            raise ValueError(
                "X must have the same number of features as during fit."
            )
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values.")

        hidden_values = self._activation_instance.forward(
            self._hidden.forward(X)
        )
        logits = self._output.forward(hidden_values)
        return self._output_activation.forward(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the class label with highest probability per sample.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            A one-dimensional array of shape (n_samples,) with predicted
            classes.
        """
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        return self._classes[class_indices]
