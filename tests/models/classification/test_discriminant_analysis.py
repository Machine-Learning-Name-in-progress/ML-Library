# tests/test_discriminant_analysis.py
import numpy as np
import pytest
from ml_library.models.classification.discriminant_analysis import QDA, LDA

def test__qda_invalid_reg_param_raises():
    with pytest.raises(ValueError):
        QDA(-1)


def test__lda_invalid_reg_param_raises():
    with pytest.raises(ValueError):
        LDA(-1)

def test_qda_priors_before_fit_raises():
    with pytest.raises(RuntimeError):
        QDA().priors

def test_lda_priors_before_fit_raises():
    with pytest.raises(RuntimeError):
        LDA().priors

def test_qda_means_before_fit_raises():
    with pytest.raises(RuntimeError):
        QDA().means

def test_lda_means_before_fit_raises():
    with pytest.raises(RuntimeError):
        LDA().means

def test_qda_cov_before_fit_raises():
    with pytest.raises(RuntimeError):
        QDA().covariances

def test_lda_cov_before_fit_raises():
    with pytest.raises(RuntimeError):
        LDA().covariances

def test_coef_before_fit_raises():
    with pytest.raises(RuntimeError):
        LDA().coef

def test_intercept_before_fit_raises():
    with pytest.raises(RuntimeError):
        LDA().intercept

X = np.array([0, 2, 10, 11, 12]).reshape(-1, 1)
Y = np.array(["cat", "cat", "dog", "dog", "dog"])

# Values were computed by hand since it's simple enough.
def test_qda_fitted_parameters():
    qda = QDA()
    qda.fit(X,Y)

    np.testing.assert_allclose(qda.priors, np.array([2/5, 3/5]))
    np.testing.assert_allclose(qda.means, np.array([1, 11]).reshape(-1, 1))
    np.testing.assert_allclose(qda.covariances, np.array([2,1]).reshape(2,1,1))

def test_lda_fitted_parameters():
    lda = LDA()
    lda.fit(X,Y)

    np.testing.assert_allclose(lda.coef, np.array([[3/4],[33/4]]))
    np.testing.assert_allclose(lda.intercept, np.array([-3/8 + np.log(2/5), -363/8 + np.log(3/5)]))


def test_qda_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        QDA().predict(X)

def test_lda_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        LDA().predict(X)

def test_qda_predict():
    qda = QDA()
    qda.fit(X, Y)
    np.testing.assert_array_equal(qda.predict(X), Y)

def test_lda_predict():
    lda = LDA()
    lda.fit(X, Y)
    np.testing.assert_array_equal(lda.predict(X), Y)

def test_qda_predict_probabilities():
    qda = QDA()
    qda.fit(X, Y)
    proba = qda.predict_probabilities(X)
    assert proba.shape == (5, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1)

def test_lda_predict_probabilities():
    lda = LDA()
    lda.fit(X, Y)
    proba = lda.predict_probabilities(X)
    assert proba.shape == (5, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1)

def test_qda_decision():
    qda = QDA()
    qda.fit(X, Y)
    scores = qda.decision(X)
    assert scores.shape == (5, 2)
    np.testing.assert_array_equal(np.argmax(scores, axis=1), [0, 0, 1, 1, 1])

def test_lda_decision():
    lda = LDA()
    lda.fit(X, Y)
    scores = lda.decision(X)
    assert scores.shape == (5, 2)
    np.testing.assert_array_equal(np.argmax(scores, axis=1), [0, 0, 1, 1, 1])

# Computed by hand: pooled variance 4/3, boundary at 6 + log(2/3)/7.5
def test_lda_decision_boundary():
    lda = LDA()
    lda.fit(X, Y)
    boundary = 6 + np.log(2/3) / 7.5
    np.testing.assert_allclose(lda.predict_probabilities(np.array([[boundary]])), [[0.5, 0.5]])

def test_qda_predict_wrong_number_of_features_raises():
    X = np.array([[0, 0], [2, 1], [0, 2], [10, 10], [11, 12], [12, 11]])
    Y = np.array(["cat", "cat", "cat", "dog", "dog", "dog"])
    qda = QDA()
    qda.fit(X, Y)
    with pytest.raises(ValueError):
        qda.predict(X[:, :1])