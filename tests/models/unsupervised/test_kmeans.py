# tests/test_kmeans.py
import numpy as np
import pytest
from ml_library.utils import _random
from ml_library.models.unsupervised.kmeans import Kmeans

# todo: put in a global module
tolerance = 1e-6

def test_centroids_before_fit_raises():
    with pytest.raises(RuntimeError):
        Kmeans(2).centroids

def test_cluster_before_fit_raises():
    with pytest.raises(RuntimeError):
        Kmeans(2).clusters

def test_k_less_than_1_raise():
    with pytest.raises(ValueError):
        Kmeans(0)

def test_k1_centroid_is_mean():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 6.0]])
    km = Kmeans(1, seed=0)
    km.fit(X)
    np.testing.assert_allclose(km.centroids[0], X.mean(axis=0))

def test_kn_centroids_is_points():
    rng = _random.get_rng(0)
    X = rng.normal(size=(10, 2))
    km = Kmeans(10, seed=0).fit(X)

    assert len(set(km.clusters)) == 10 # every point in its own cluster
    np.testing.assert_allclose(km.centroids[km.clusters], X) # each point's centroid is itself

def test_k_greater_than_n_raise():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 6.0]])
    km = Kmeans(4)
    with pytest.raises(ValueError):
        km.fit(X)

def test_1D_array_raise():
    X = np.array([1,2,3])
    with pytest.raises(ValueError):
        Kmeans(2, seed=0).fit(X)

def test_wrong_dim_predict_raise():
    X = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 6.0]])
    km = Kmeans(2).fit(X)
    p = np.array([[1,2,3]])
    with pytest.raises(ValueError):
        km.predict(p)

def test_max_iter_warn():
    rng = _random.get_rng(0)
    X = rng.normal(size=(20,3))
    with pytest.warns(UserWarning):
        Kmeans(5, seed=0, max_iter=1).fit(X)

@pytest.mark.parametrize("seed", range(10)) #Tests for seed = 0,1,2,...,9
def test_converged_fit_is_fixed_point(seed):
    X = _random.get_rng(seed).normal(size=(200, 2))
    km = Kmeans(4, seed=seed, max_iter=1000).fit(X)

    # every point is assigned to its nearest centroid
    d = np.linalg.norm(X[:, None] - km.centroids[None], axis=2)
    np.testing.assert_array_equal(km.clusters, d.argmin(axis=1))
    # every non-empty cluster's centroid is its mean
    for j in np.unique(km.clusters):
        np.testing.assert_allclose(km.centroids[j], X[km.clusters == j].mean(axis=0))
    # stored labels agree with predict
    np.testing.assert_array_equal(km.predict(X), km.clusters)

def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True if two labelings define the same clusters, up to renaming the labels."""
    pairs = set(zip(a.tolist(), b.tolist()))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))

# todo: center the clusters before calculating the mean() and then move offset them back to original
@pytest.mark.xfail(reason="distance expansion loses precision far from the origin; needs centering")
@pytest.mark.parametrize("seed", range(5))
def test_translation_invariance(seed):
    X = _random.get_rng(seed).normal(size=(200, 2))
    shift = 1e8

    a = Kmeans(4, seed=seed, max_iter=1000).fit(X)
    b = Kmeans(4, seed=seed, max_iter=1000).fit(X + shift)

    assert same_partition(a.clusters, b.clusters)
    np.testing.assert_allclose(b.centroids - shift, a.centroids, atol=tolerance)