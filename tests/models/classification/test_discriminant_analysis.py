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

# TODO: add dimensonality test

# Values were computed by hand since it's simple enough.
def test_qda_fitted_parameters():
    X = np.array([0, 2, 10, 11, 12]).reshape(-1,1)
    Y = np.array(["cat", "cat", "dog", "dog", "dog"])
    qda = QDA()
    qda.fit(X,Y)

    np.testing.assert_allclose(qda.priors, np.array([2/5, 3/5]))
    np.testing.assert_allclose(qda.means, np.array([1, 11]).reshape(-1, 1))
    np.testing.assert_allclose(qda.covariances, np.array([2,1]).reshape(2,1,1))

def test_lda_fitted_parameters():
    X = np.array([0, 2, 10, 11, 12]).reshape(-1,1)
    Y = np.array(["cat", "cat", "dog", "dog", "dog"])
    lda = LDA()
    lda.fit(X,Y)

    np.testing.assert_allclose(lda.coef, np.array([[3/4],[33/4]]))
    np.testing.assert_allclose(lda.intercept, np.array([-3/8 + np.log(2/5), -363/8 + np.log(3/5)]))

