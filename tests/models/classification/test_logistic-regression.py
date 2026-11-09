import numpy as np
import pytest

from ml_library.models.classification.logistic_regression import logistic_regression


@pytest.fixture
def data():
    x = np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]])
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    return x, y


def test_logistic_regression_coefficients(data):
    x, y = data
    model = LogisticRegression()
    model.fit(x, y)
    assert model._beta == pytest.approx([np.log(1 / 3), 2 * np.log(3)], abs=1e-3)


def test_logistic_regression_probabilities_sum_to_one(data):
    x, y = data
    model = LogisticRegression()
    model.fit(x, y)
    probs = model.predict_probabilities(x)
    assert probs.shape == (8, 2)
    assert probs.sum(axis=1) == pytest.approx(np.ones(8))


@pytest.mark.parametrize(
    "labels, expected",
    [
        (["no", "yes"], ["no", "yes"]),
        ([-1, 1], [-1, 1]),
        ([3, 7], [3, 7]),
    ],
)
def test_logistic_regression_any_labels(data, labels, expected):
    x, y = data
    y_labels = np.array(labels)[y]
    model = LogisticRegression()
    model.fit(x, y_labels)
    np.testing.assert_array_equal(model.predict(np.array([[0.0], [1.0]])), expected)


def test_logistic_regression_separable_data_warns():
    model = LogisticRegression()
    with pytest.warns(
        RuntimeWarning, match="Did not converge in max_iter = 5000 iterations"
    ):
        model.fit(np.array([[0.0], [1.0]]), np.array([0, 1]))


def test_logistic_regression_fit_exceptions(data):
    x, y = data
    model = LogisticRegression()
    with pytest.raises(ValueError) as excep:
        model.fit(x, y.reshape(-1, 1))
    assert str(excep.value) == "y must be a 1D array with shape (n_samples)"
    with pytest.raises(ValueError) as excep:
        model.fit(x, np.zeros(8))
    assert str(excep.value) == "Logistic regression needs exactly 2 classes (1 given)"


def test_logistic_regression_bool_and_numpy_params():
    with pytest.raises(TypeError):
        LogisticRegression(learn_rate=True)
    with pytest.raises(TypeError):
        LogisticRegression(max_iter=True)
    with pytest.raises(TypeError):
        LogisticRegression(tol=False)
    LogisticRegression(
        learn_rate=np.float32(0.1), max_iter=np.int64(10), tol=np.float64(1e-3)
    )
