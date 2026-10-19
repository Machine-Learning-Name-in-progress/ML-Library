"""Grid search cross validation for model hyperparameter tuning."""

import itertools

import numpy as np

from ml_library.model_selection.kfold import kfold_indices


def grid_search_cv(model, param_grid, X, y, k=5, scoring=None):
    """
    Grid search method for tuning a model's hyperparameters.

    Parameters:
    model (class): The model class to use.
    param_grid (dict): The parameter grid to use.
    X (array-like): The input features.
    y (array-like): The target values.
    k (int, optional): The number of folds.
    scoring (callable, optional): The scoring function.

    Returns:
    dict: A dictionary containing the best parameters,
        best score, and best estimator.
    """
    if scoring is None:
        # uses standart MSE (pre-made MSE is not ready in ML-Library)
        def scoring(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

    n = len(X)
    splits = kfold_indices(n, k)

    # all hyperparameters combinations
    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = [
        dict(zip(param_keys, p)) for p in itertools.product(*param_values)
    ]

    best_score = np.inf
    best_params = None

    for params in param_combinations:
        fold_scores = []
        for train_index, val_index in splits:
            estimator = model(**params)
            estimator.fit(X[train_index], y[train_index])

            y_pred = estimator.predict(X[val_index])

            fold_scores.append(scoring(y[val_index], y_pred))

        mean_score = np.mean(fold_scores)

        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    best_estimator = model(**best_params)
    best_estimator.fit(X, y)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_estimator": best_estimator,
    }
