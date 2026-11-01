import numpy as np
import pytest
from ml_library.models.unsupervised.pca import pca

# TODO: unify random seeds
# TODO: add more tests
np.random.seed(67)
atol = 1e-6

def test_pca_invalid_solver():
    pass

def test_pca_eigen():
    pass

def test_pca_svd():
    pass

def test_pca_standardization_constant_feature():
    pass

def test_pca_invalid_component_number():
    with pytest.raises(ValueError):
        pca(n_components=-1)

    with pytest.raises(TypeError):
        pca(n_components=0.5)

    with pytest.raises(TypeError):
        pca(n_components="67")
    
    with pytest.raises(TypeError):
        pca(n_components="string")

def test_pca_array_dimensions():
    pass

def test_pca_less_features_than_components():
    pass

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