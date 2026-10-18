"""Bootstrap methods to select the best model."""

from ast import Continue

import numpy as np


def bootstrap_sample(X, y, seed=None):
    """Generate bootstrap samples."""
    if seed:
        np.random.seed(seed)
    N = X.shape[0]

    indexes = np.random.choice(N, N, replace=True)

    return X[indexes], y[indexes]


def bootstrap_ci(model, X, y, n_bootstraps=1000, alpha=0.05, seed=None):
    """Generate bootstrap confidence intervals."""
    if seed:
        np.random.seed(seed)

    # initiates an empty array to receive the estimatives
    L = np.empty(n_bootstraps)
    for b in range(n_bootstraps):
        X_b, y_b = bootstrap_sample(X, y)
        model.fit(X_b, y_b)
        L[b] = model.predict(X).mean()

    std_err = np.std(L, ddof=1)  # ddof = 1 for sampled std
    ci_lower = np.percentile(L, (alpha / 2) * 100)
    ci_upper = np.percentile(L, (1 - alpha / 2) * 100)

    return {"std_err": std_err, "ci_lower": ci_lower, "ci_upper": ci_upper}
