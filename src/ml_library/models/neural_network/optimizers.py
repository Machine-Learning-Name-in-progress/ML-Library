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

            # update layer bises -> b = b - lr * dL/db
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


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the Adam optimizer.

        Args:
            lr: learning rate.
            beta1: decay rate for first moment.
            beta2: decay rate for second moment.
            eps: small constant to avoid division by zero.
        """
        super().__init__(lr=lr)
        if not (0.0 <= beta1 < 1.0) or not (0.0 <= beta2 < 1.0):
            raise ValueError("Betas must be in range [0, 1).")
        if eps <= 0.0:
            raise ValueError("Epsilon must be positive.")

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = {}

    def update(
        self,
        param: np.ndarray,
        grad: np.ndarray,
        key: str | None = None,
    ) -> np.ndarray:
        """Compute and apply the Adam update to the parameters.

        Args:
            param: Current parameters values.
            grad: Gradient with respect to parameters.
            key: Identifier used to store moments.

        Returns:
            Updated parameter array.
        """
        if param.shape != grad.shape:
            raise ValueError("Param and grad shapes must match.")

        if key is None:
            key = str(id(param))

        # initialize moments with zeros on first step -> m_0 = 0, v_0 = 0
        if key not in self.m:
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)
            self.t[key] = 0

        self.t[key] += 1
        t = self.t[key]

        # update first moment-> m_t = beta1 * m_t-1 + (1 - beta1) * grad
        self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grad

        # update second moment -> v_t = beta2 * v_t-1 + (1 - beta2) * grad^2
        self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (grad**2)

        # bias correction -> m_hat = m_t / (1 - beta1^t)
        m_hat = self.m[key] / (1.0 - self.beta1**t)
        v_hat = self.v[key] / (1.0 - self.beta2**t)

        # update parameters -> w = w - lr * m_hat / (sqrt(v_hat) + eps)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """AdamW optimizer with decoupled weight decay."""

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """Initialize the AdamW optimizer.

        Args:
            lr: learning rate.
            beta1: decay rate for first moment.
            beta2: decay rate for second moment.
            eps: small constant to avoid division by zero.
            weight_decay: weight decay factor.
        """
        super().__init__(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        if weight_decay < 0.0:
            raise ValueError("Weight decay must be non-negative.")

        self.weight_decay = weight_decay

    def update(
        self,
        param: np.ndarray,
        grad: np.ndarray,
        key: str | None = None,
    ) -> np.ndarray:
        """Compute and apply the AdamW update to the parameters.

        Args:
            param: Current parameters values.
            grad: Gradient with respect to parameters.
            key: Identifier used to store moments.

        Returns:
            Updated parameter array.
        """
        if param.shape != grad.shape:
            raise ValueError("Param and grad shapes must match.")

        # appli wiight decay only to weights -> w = w - lr * lambda * w
        if key is not None and "bias" in key:
            decayed_param = param
        else:
            decayed_param = param - self.lr * self.weight_decay * param

        return super().update(decayed_param, grad, key=key)
