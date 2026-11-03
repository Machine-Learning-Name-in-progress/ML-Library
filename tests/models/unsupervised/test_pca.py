import numpy as np
import pytest
from ml_library.models.unsupervised.pca import pca

# TODO: unify random seeds globally
np.random.seed(67)
atol = 1e-6

def test_pca_invalid_solver():
    with pytest.raises(ValueError, match="solver"):
        pca(n_components=2, solver="guess")

def test_pca_methods():
    for solver_choice in ['eigen', 'svd']:
        X = np.random.rand(20, 4)
        pca_test = pca(n_components=2, solver=solver_choice)
        X_reduced = pca_test.fit_transform(X)

        assert pca_test.components.shape == (4, 2)
        assert X_reduced.shape == (20, 2)

def test_pca_eigen_svd_consistency():
    X = np.random.rand(42, 6)

    X_eigen = pca(n_components=3, solver='eigen')
    X_svd = pca(n_components=3, solver='svd')

    X_eigen_reduced = X_eigen.fit_transform(X)
    X_svd_reduced = X_svd.fit_transform(X)

    assert np.abs(X_eigen_reduced) == pytest.approx(np.abs(X_svd_reduced), abs=atol)

def test_pca_standardization():
    X = np.array([
        [1.0, 2000.0, 0.004],
        [1.0, 4000.0, 0.005],
        [1.0, 6000.0, 0.006]
    ])
    pca_test = pca(n_components=2, standardize=True)
    pca_test.fit(X)

    assert pca_test.std[0] == pytest.approx(0.0, abs=atol)
    assert pca_test.std[1] == pytest.approx(np.sqrt(8/3) * 1000, abs=atol)
    assert pca_test.std[2] == pytest.approx(np.sqrt(2/3) / 1000, abs=atol)

    with pytest.raises(TypeError, match="must be a boolean"):
        pca(n_components=2, standardize="Yes, please.")

def test_pca_input():
    pca_test = pca(n_components=1)
    X1 = [[1, 2], [3, 4]]
    X2 = "42"
    X3 = np.array([["abc", 67], [42, "123"]])

    X_reduced = pca_test.fit_transform(X1)
    assert isinstance(X_reduced, np.ndarray)

    with pytest.raises(TypeError, match="must be array-like"):
        pca_test.fit(X2)

    with pytest.raises(TypeError, match="contain only numerical entries"):
        pca_test.fit(X3)

def test_pca_invalid_component_number():
    with pytest.raises(ValueError, match="must be positive"):
        pca(n_components=-1)

    with pytest.raises(TypeError, match="must be an integer, got float"):
        pca(n_components=0.5)

    with pytest.raises(TypeError, match="must be an integer, got str"):
        pca(n_components="67")
    
    with pytest.raises(TypeError, match="must be an integer, got str"):
        pca(n_components="string")

def test_pca_array_dimensions():
    pca_test = pca(n_components=2)
    array1 = np.array([42, 21, 67])
    array3 = np.random.rand(3, 3, 3)

    with pytest.raises(ValueError, match="Expected a 2-D array"):
        pca_test.fit(array1)

    with pytest.raises(ValueError, match="Expected a 2-D array"):
        pca_test.fit(array3)

def test_pca_less_features_than_components():
    X = np.random.rand(42, 2)
    pca_test = pca(n_components=3)

    with pytest.raises(ValueError):
        pca_test.fit(X)

def test_pca_transform_before_fit():
    X = np.random.rand(8, 3)
    pca_test = pca(n_components=2)

    with pytest.raises(RuntimeError):
        pca_test.transform(X)

def test_pca_orthogonal_components():
    X = np.random.rand(42, 4)
    pca_test = pca(n_components=2)
    pca_test.fit(X)

    dot_prod = np.dot(pca_test.components[:, 0], pca_test.components[:, 1])
    assert dot_prod == pytest.approx(0.0, abs=atol)