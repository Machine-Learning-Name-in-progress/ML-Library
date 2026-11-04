import numpy as np
import pytest

from ml_library.models.classification.logistic_regression import logistic_regression


@pytest.fixture
def data():
    x = np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]])
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    return x, y


def test_logistic_regression_probabilities(data):
    x, y = data

    model = logistic_regression()
    model.fit(x, y)

    probs = model.predict_probabilities(np.array([[0.0], [1.0]]))

    assert probs[:, 1] == pytest.approx([1 / 4, 3 / 4], abs=1e-4)
    assert probs[:, 0] == pytest.approx([3 / 4, 1 / 4], abs=1e-4)


def test_logistic_regression_predict(data):
    x, y = data

    model = logistic_regression()
    model.fit(x, y)
    np.testing.assert_array_equal(model.predict(np.array([[1.0], [0.0]])), [1, 0])


def test_logistic_regression_separable_data_warns():
    model = logistic_regression(max_iter=100)
    with pytest.warns(RuntimeWarning):
        model.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))


def test_logistic_regression_exceptions(data):
    x, y = data

    with pytest.raises(TypeError) as excep:
        logistic_regression(learn_rate="0.5")
    assert str(excep.value) == "'learn_rate' must be a number"

    with pytest.raises(ValueError) as excep:
        logistic_regression(learn_rate=0)
    assert str(excep.value) == "'learn_rate' must be greater than 0"

    with pytest.raises(TypeError) as excep:
        logistic_regression(max_iter=10.5)
    assert str(excep.value) == "'max_iter' must be an integer"

    with pytest.raises(ValueError) as excep:
        logistic_regression(max_iter=0)
    assert str(excep.value) == "'max_iter' must be greater than 0"

    with pytest.raises(TypeError) as excep:
        logistic_regression(tol="0.1")
    assert str(excep.value) == "'tol' must be a number"

    with pytest.raises(ValueError) as excep:
        logistic_regression(tol=-1)
    assert str(excep.value) == "'tol' must be greater than 0"

    model = logistic_regression()

    with pytest.raises(ValueError) as excep:
        model.fit(x, np.array([0, 0, 0, 1, 0, 1, 1, 2]))
    assert str(excep.value) == "Logistic regression only works with two classes 0/1"

    with pytest.raises(ValueError) as excep:
        model.fit(x, y[:3])
    assert (
        str(excep.value) == "Each point from x needs a result in y: same number of rows"
    )

    with pytest.raises(ValueError) as excep:
        model.predict(x)
    assert str(excep.value) == "Cannot predict before training. Use '.fit()' first"
