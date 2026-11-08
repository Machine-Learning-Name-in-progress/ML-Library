"""Bootstrap methods to select the best model."""

import numpy as np
from ml_library.utils._random import get_rng


def bootstrap_sample(X, y, seed=None):
    """Generate bootstrap samples.

    Args:
        X (array-like): The input features.
        y (array-like): The target values.
        seed (int, optional): The seed for the random number generator.

    Returns:
        tuple: (X_sample, y_sample) containing the bootstrap samples.
    """
    rng = get_rng(seed)
    n_samples = X.shape[0]

    indexes = rng.choice(n_samples, size=n_samples, replace=True)

    return X[indexes], y[indexes]


def bootstrap_ci(model, X, y, n_bootstraps=1000, alpha=0.05, seed=None):
    """Generate bootstrap confidence intervals.

    Args:
        model (object): The model to use.
        X (array-like): The input features.
        y (array-like): The target values.
        n_bootstraps (int, optional): The number of bootstrap samples.
        alpha (float, optional): The significance level.
        seed (int, optional): The seed for the random number generator.

    Returns:
        dict: A dictionary containing the standard error and confidence
            interval ('std_err', 'ci_lower', 'ci_upper').
    """
    rng = get_rng(seed)

    # initiates an empty array to receive the estimates
    estimates = np.empty(n_bootstraps)
    for b in range(n_bootstraps):
        X_b, y_b = bootstrap_sample(X, y, seed=rng)
        model.fit(X_b, y_b)
        estimates[b] = model.predict(X).mean()

    std_err = np.std(estimates, ddof=1)  # ddof = 1 for sample std
    ci_lower = np.percentile(estimates, (alpha / 2) * 100)
    ci_upper = np.percentile(estimates, (1 - alpha / 2) * 100)

    return {"std_err": std_err, "ci_lower": ci_lower, "ci_upper": ci_upper}
