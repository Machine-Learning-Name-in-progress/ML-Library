"""Optimizers classes for neural networks."""

from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    """Base class for neural network optimizers."""

    def __init__(self, lr: float = 0.01) -> None:
        """Initialize the optimizer.

        Args:
            lr: Learning rate. Must be positive.
        """
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("Learning rate must be finite and positive.")
        self.lr = lr

    @abstractmethod
    def update(
        self, param: np.ndarray, grad: np.ndarray, key: str | None = None
    ) -> np.ndarray:
        """Update a parameters array given its gradient."""

    def step(self, layers: list) -> None:
        """Update weights and biases for a list of network layers.

        Args:
            layers: Single layer or list of layers from the neural network.
        """
        if not isinstance(layers, list):
            layers = [layers]

        for i, layer in enumerate(layers):
            # update layer weights -> w = w - lr * dL/dw
            if (
                hasattr(layer, "weights")
                and getattr(layer, "grad_weights", None) is not None
            ):
                layer.weights = self.update(
                    layer.weights,
                    layer.grad_weights,
                    key=f"layer_{i}_weights",
                )

            # update layer biases -> b = b - lr * dL/db
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
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, lr: float = 0.01, momentum: float = 0.0) -> None:
        """Initialize the SGD.

        Args:
            lr: Learning rate.
            momentum: Momentum factor in range [0, 1).
        """
        super().__init__(lr=lr)
        if not (0.0 <= momentum < 1.0):
            raise ValueError("Momentum must be in range [0, 1).")

        self.momentum = momentum
        self._velocities = {}

    def update(
        self,
        param: np.ndarray,
        grad: np.ndarray,
        key: str | None = None,
    ) -> np.ndarray:
        """Compute and apply the SGD update to the parameters.

        Args:
            param: Current parameters values.
            grad: Gradient with respect to parameters.
            key: Identifier used to store the velocity.

        Returns:
            Updated parameter array.
        """
        if param.shape != grad.shape:
            raise ValueError("Param and grad shapes must match.")

        # standard gradient descendent step -> w = w - lr * grad
        if self.momentum == 0.0:
            return param - self.lr * grad

        if key is None:
            key = "default"

        # initialize velocity with zeros on first step -> v_0 = 0
        if key not in self._velocities:
            self._velocities[key] = np.zeros_like(param, dtype=np.float64)

        # accumulate momentum velocity -> v_t = momentum * v_t-1 + lr * grad
        self._velocities[key] = (
            self.momentum * self._velocities[key] + self.lr * grad
        )

        # update parameters -> w = w - v_t
        return param - self._velocities[key]


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
            lr: Learning rate.
            beta1: Decay rate for first moment.
            beta2: Decay rate for second moment.
            eps: Small constant to avoid division by zero.
        """
        super().__init__(lr=lr)
        if not (0.0 <= beta1 < 1.0) or not (0.0 <= beta2 < 1.0):
            raise ValueError("Betas must be in range [0, 1).")
        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError("Epsilon must be finite and positive.")

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = {}
        self._v = {}
        self._t = {}

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
            key = "default"

        # initialize moments with zeros on first step -> m_0 = 0, v_0 = 0
        if key not in self._m:
            self._m[key] = np.zeros_like(param, dtype=np.float64)
            self._v[key] = np.zeros_like(param, dtype=np.float64)
            self._t[key] = 0

        self._t[key] += 1
        t = self._t[key]

        # update first moment-> m_t = beta1 * m_t-1 + (1 - beta1) * grad
        self._m[key] = self.beta1 * self._m[key] + (1.0 - self.beta1) * grad

        # update second moment -> v_t = beta2 * v_t-1 + (1 - beta2) * grad^2
        self._v[key] = self.beta2 * self._v[key] + (1.0 - self.beta2) * (grad**2)

        # bias correction -> m_hat = m_t / (1 - beta1^t)
        m_hat = self._m[key] / (1.0 - self.beta1**t)
        v_hat = self._v[key] / (1.0 - self.beta2**t)

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
            lr: Learning rate.
            beta1: Decay rate for first moment.
            beta2: Decay rate for second moment.
            eps: Small constant to avoid division by zero.
            weight_decay: Weight decay factor.
        """
        super().__init__(lr=lr, beta1=beta1, beta2=beta2, eps=eps)

        if weight_decay < 0.0 or not np.isfinite(weight_decay):
            raise ValueError("Weight decay must be non-negative and finite.")

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

        # apply weight decay only to weights -> w = w - lr * lambda * w
        if key is not None and "bias" in key:
            decayed_param = param
        else:
            decayed_param = param - self.lr * self.weight_decay * param

        return super().update(decayed_param, grad, key=key)
