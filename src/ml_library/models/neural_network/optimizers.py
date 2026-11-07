"""Optimizers classes for neural notworks"""

from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    """Base class for neiural network optimizers."""

    def __init__(self, lr: float = 0.01) -> None:
        """Initialize the optimizer.

        Args:
            lr: learning rate. Must be positive.
        """
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive")
        self.lr = lr

    @abstractmethod
    def update(
        self, param: np.ndarray, grad: np.ndarray, key: str | None = None
    ) -> np.ndarray:
        """Update a parameters array given its gradient."""

    def step(self, layers: list) -> None:
        """Update  weights and baises for a list of network layers.

        Args:
            layers: SIngle layer or list of layers from the neural network.
        """
        if not isinstance(layers, (list, tuple)):
            layers = [layers]

        for i, layer in enumerate(layers):
            # update layer wights -> w = w - lr * dL/dw
            if (
                hasattr(layer, "weights")
                and getattr(layer, "grad_weights", None) is not None
            ):
                layer.weights = self.update(
                    layer.weights,
                    layer.grad_weights,
                    key=f"layer_{i}_weights",
                )

            # update layer baises -> b = b - lr * dL/db
            if (
                hasattr(layer, "bias")
                and getattr(layer, "grad_bias", None) is not None
            ):
                layer.bias = self.update(
                    layer.bias,
                    layer.grad_bias,
                    key=f"layer_{i}_bias",
                )


class SGD(Optimizer):
    """Stochasctic Gradient Descent optimizer."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.0) -> None:
        """Initialzie the SGD.

        Args:
            lr: learning rate.
            momentum: momentumfactor in range [0, 1).
        """
        super().__init__(lr=lr)
        if momentum < 0 or momentum >= 1.0:
            raise ValueError("Momentum must be in range [0, 1).")

        self.momentum = momentum
        self.velocities = {}

    def update(
        self,
        param: np.ndarray,
        grad: np.ndarray,
        key: str | None = None,
    ) -> np.ndarray:
        """Compute and apply the SGD update to the parameters.

        Args:
            param: Current parametersvalues.
            grad: Gradient with respect to parameters.
            key: Identifier used to store the velocity.

        Returns:
            Updated parameter array.
        """
        if param.shape != grad.shape:
            raise ValueError("Param and grad shapes must match..")

        # standard gradient descendent step -> w = w - lr * grad
        if self.momentum == 0.0:
            return param - self.lr * grad

        if key is None:
            key = str(id(param))

        # initialize velocity with zerps on firts step -> v_0 = 0
        if key not in self.velocities:
            self.velocities[key] = np.zeros_like(param)

        # accumulte momentum velocity -> v_t = momentum * v_t-1 + lr * grad
        self.velocities[key] = (
            self.momentum * self.velocities[key] + self.lr * grad
        )

        # update parameters -> w = w - v_t
        return param - self.velocities[key]
