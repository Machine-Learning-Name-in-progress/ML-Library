import numpy as np
import pytest

from ml_library.models.classification.knn import KNN


@pytest.fixture
def data():
    x = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
    y = np.array([0, 0, 1, 1, 1])
    return x, y


def test_knn_predict(data):
    x, y = data

    x_new = np.array([[1.4], [10.4]])

    model = KNN(1)
    model.fit(x, y)
    np.testing.assert_array_equal(model.predict(x_new), [0, 1])

    model = KNN(3)
    model.fit(x, y)
    np.testing.assert_array_equal(model.predict(x_new), [0, 1])

    model = KNN(5)
    model.fit(x, y)
    np.testing.assert_array_equal(model.predict(x_new), [1, 1])


def test_knn_k1_returns_training_labels(data):
    x, y = data

    model = KNN(1)
    model.fit(x, y)
    np.testing.assert_array_equal(model.predict(x), y)


def test_knn_tie_chooses_smallest_label():
    model = KNN(2)
    model.fit(np.array([[0.0], [2.0]]), np.array([1, 0]))
    np.testing.assert_array_equal(model.predict(np.array([[1.0]])), [0])


def test_knn_change_k(data):
    x, y = data

    changed = KNN(1)
    changed.fit(x, y)
    changed.change_K(5)

    new = KNN(5)
    new.fit(x, y)

    np.testing.assert_array_equal(changed.predict(x), new.predict(x))


def test_knn_exceptions(data):
    x, y = data

    with pytest.raises(TypeError) as excep:
        KNN(2.5)
    assert str(excep.value) == "K must be an integer"

    with pytest.raises(ValueError) as excep:
        KNN(0)
    assert str(excep.value) == "K must be an integer greater than 0"

    with pytest.raises(ValueError) as excep:
        model = KNN(1)
        model.fit(x, y[:3])
    assert str(excep.value) == "x has 5 samples while y has 3. Both need to be equal"

    with pytest.raises(ValueError) as excep:
        model = KNN(6)
        model.fit(x, y)
    assert (
        str(excep.value) == "KNN model needs at least K = 6 points to work (5 passed)"
    )

    with pytest.raises(ValueError) as excep:
        model = KNN(1)
        model.predict(x)
    assert str(excep.value) == "Cannot predict before training. Use '.fit()' first"

    with pytest.raises(ValueError) as excep:
        model = KNN(1)
        model.fit(x, y)
        model.change_K(6)
    assert (
        str(excep.value) == "KNN model needs at least K = 6 points to work (5 passed)"
    )
