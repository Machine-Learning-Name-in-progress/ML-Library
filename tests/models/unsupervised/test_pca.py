import numpy as np
import pytest
from ml_library.models.unsupervised.pca import pca

# TODO: unify random seeds
# TODO: implement more tests
np.random.seed(67)

def test_pca_invalid_component_number():
    with pytest.raises(ValueError):
        pca(n_components=-1)

    with pytest.raises(TypeError):
        pca(n_components=0.5)

    with pytest.raises(TypeError):
        pca(n_components="67")

def test_pca_array_dimensions():
    pass

def test_pca_less_features_than_components():
    pass

def test_pca_transform_before_fit():
    pass

def test_pca_orthogonal_components():
    pass