"""Test scrips for bootstrap functions."""

import numpy as np
import pytest
from ml_library.model_selection.bootstrap import bootstrap_ci, bootstrap_sample


# General arrange:
@pytest.fixture
def sample_data():
    """Fixture providing clean sample data for all the tests."""
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    return X, y


# seed is 42 by standart inside the methods, where np.random.seed(42) is setted


# bootstrap sample tests:
def test_bootstrap_sample_shape(sample_data):
    """Verify if the shape is preserved in the bootstrap samples."""
    # arrange
    X, y = sample_data
    # act
    X_boot, y_boot = bootstrap_sample(X, y)

    # assert
    np.testing.assert_array_equal(X_boot.shape, X.shape)
    np.testing.assert_array_equal(y_boot.shape, y.shape)


def test_bootstrap_sample_duplicates(sample_data):
    """Verify that bootstrap samples contain duplicates."""
    # arrange
    X, y = sample_data
    # act
    X_boot, y_boot = bootstrap_sample(X, y)

    # assert
    np.testing.assert_(len(X_boot) != len(np.unique(X_boot)))
    np.testing.assert_(len(y_boot) != len(np.unique(y_boot)))


def test_bootstrap_sample_alignment(sample_data):
    """Verify that bootstrap samples are aligned with the original data."""
    # arrange
    X, y = sample_data

    # act
    X_boot, y_boot = bootstrap_sample(X, y)

    # assert
    # every line in X_boot remains attached to its y_boot
    np.testing.assert_array_equal(X_boot[:, 0] // 2, y_boot)


def test_bootstrap_sample_reproducibility(sample_data):
    """Verify the reproducibility of bootstrap samples."""
    # arrange
    X, y = sample_data

    # act
    X_boot1, y_boot1 = bootstrap_sample(X, y, seed=42)
    X_boot2, y_boot2 = bootstrap_sample(X, y, seed=42)

    # assert
    np.testing.assert_array_equal(X_boot1, X_boot2)
    np.testing.assert_array_equal(y_boot1, y_boot2)


# bootstrap confidence interval tests:


def test_bootstrap_ci(sample_data):
    """
    Test bootstrap confidence interval method.

    Uses dummy model for quick and independent testing.
    """

    # Dummy model
    class DummyModel:
        def __init__(self):
            self._mean = None

        def fit(self, X, y):
            self._mean = np.mean(y)

        def predict(self, X):
            return np.full(len(X), self._mean)

    # arrange
    model = DummyModel()
    X, y = sample_data

    # act
    results = bootstrap_ci(model, X, y, n_bootstraps=50, alpha=0.05)

    # assert
    np.testing.assert_(isinstance(results, dict))
    np.testing.assert_equal(
        set(results.keys()), {"std_err", "ci_lower", "ci_upper"}
    )
    np.testing.assert_(results["std_err"] > 0)
    np.testing.assert_(results["ci_lower"] < results["ci_upper"])
