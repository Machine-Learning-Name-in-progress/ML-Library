"""Test module for hyperparameters grid search."""

import numpy as np
import pytest
from ml_library.model_selection.grid_search_cv import grid_search_cv


# general arrange
@pytest.fixture
def sample_data():
    """Fixture providing synthetic data for grid search tests."""
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    return X, y


class DummyRegressor:
    """Dummy Regressor for testing purposes."""

    def __init__(self, multiplier=1.0):
        """Initialize DummyRegressor."""
        self.multiplier = multiplier
        self.learned_mean = None

    def fit(self, X, y):
        """Fit DummyRegressor."""
        self.learned_mean = np.mean(y)
        return self

    def predict(self, X):
        """Predict with DummyRegressor."""
        return np.full(len(X), self.learned_mean * self.multiplier)


def test_grid_search_cv_finds_best_params(sample_data):
    """Test if grid search correctly identifies the best parameters."""
    # arrange
    X, y = sample_data
    param_grid = {"multiplier": [1.0, 2.0, 3.0]}

    # act
    results = grid_search_cv(DummyRegressor, param_grid, X, y, k=5)

    # assert dict
    np.testing.assert_(isinstance(results, dict))
    np.testing.assert_(
        set(results.keys()), {"best_params", "best_score", "best_estimator"}
    )

    # assert best
    np.testing.assert_equal(results["best_params"], {"multiplier": 1.0})

    # assert score
    np.testing.assert_(results["best_score"] > 0)


def test_grid_search_cv_refit(sample_data):
    """Test if grid search correctly refits the model."""
    # arrange
    X, y = sample_data
    param_grid = {"multiplier": [1.0, 2.0, 3.0]}

    # act
    results = grid_search_cv(DummyRegressor, param_grid, X, y, k=5)
    best_model = results["best_estimator"]

    # assert instance
    np.testing.assert_(isinstance(best_model, DummyRegressor))

    # assert best parameter
    np.testing.assert_equal(best_model.multiplier, 1.0)

    # assert refit
    np.testing.assert_almost_equal(
        best_model.learned_mean, np.mean(y), decimal=7
    )

    predictions = best_model.predict(X)
    # assert predictions shape
    np.testing.assert_equal(predictions.shape, y.shape)
