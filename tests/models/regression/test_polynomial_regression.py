import pytest
import numpy as np
from polynomial_regression import polynomial_regression

# TODO: unify global seeds
np.random.seed(67)
atol = 1e-6

def test_poly_design_matrix():
    poly_test = polynomial_regression(degree=3)
    x = np.array([2.0, 5.0])
    X = poly_test._build_design(x)

    X_expec = np.array([
                        [2.0, 4.0, 8.0], 
                        [5.0, 25.0, 125.0]
                    ])

    assert X == pytest.approx(X_expec, abs=atol)

def test_poly_examples():
    x_test = np.array([-2, -1, 0, 1, 2])
    y_test = 3 + 2*x_test + x_test**2
    noise = np.random.normal(0.0, 0.1, size=len(x_test))
    y_noise = y_test + noise
    deg = 4

    test_ols = polynomial_regression(degree=deg, model_choice='ols', intercept=True)
    test_ols.fit(x_test, y_test)
    assert test_ols.coefficients == pytest.approx([3.0, 2.0, 1.0, 0.0, 0.0], abs=atol)

    test_ols2 = polynomial_regression(degree=deg, model_choice='ols', intercept=True)
    test_ols2.fit(x_test, y_noise)
    assert test_ols2.coefficients == pytest.approx([3.0, 2.0, 1.0, 0.0, 0.0], abs=0.30)

    test_ridge = polynomial_regression(degree=deg, model_choice='ridge', intercept=True, lam=0.1)
    test_ridge.fit(x_test, y_noise)
    assert test_ridge.coefficients == pytest.approx([3.0, 2.0, 1.0, 0.0, 0.0], abs=0.25)

    test_lasso = polynomial_regression(degree=deg, model_choice='lasso', intercept=True, lam=0.1)
    test_lasso.fit(x_test, y_noise)
    assert test_lasso.coefficients == pytest.approx([3.0, 2.0, 1.0, 0.0, 0.0], abs=0.1)

def test_poly_init_exceptions():
    with pytest.raises(TypeError, match="int data type"):
        polynomial_regression(degree=6.7)

    with pytest.raises(ValueError, match="positive"):
        polynomial_regression(degree=-2)

    with pytest.raises(ValueError, match="Model choice"):
        polynomial_regression(degree=2, model_choice=67)

    with pytest.raises(ValueError, match="Model choice"):
        polynomial_regression(degree=2, model_choice="guess")

    with pytest.raises(TypeError, match="bool data type"):
        polynomial_regression(degree=2, intercept="Yes, please.")

    with pytest.raises(TypeError, match="float or int"):
        polynomial_regression(degree=2, lam="67")

    with pytest.raises(ValueError, match="positive"):
        polynomial_regression(degree=2, lam=-0.5)

    with pytest.raises(TypeError, match="float or int"):
        polynomial_regression(degree=2, eps="67")

    with pytest.raises(ValueError, match="positive"):
        polynomial_regression(degree=2, eps=-42)

    with pytest.raises(TypeError, match="int data type"):
        polynomial_regression(degree=2, max_iter="67")

    with pytest.raises(ValueError, match="positive"):
        polynomial_regression(degree=2, max_iter=-42)

def test_poly_data_exceptions():
    test_poly = polynomial_regression(degree=3)
    x = np.array([1, 2, 3, 4])
    y = np.array([5, 6, 7, 8])
    array_2d = np.array([[1, 2], [3, 4]])
    array_mismatch = np.array([5, 6, 7])
    x_type = np.array(["a", "b", "c"])
    y_type = np.array(["d", "e", "f"])
    x_type2 = "string_x"
    y_type2 = "string_y"

    with pytest.raises(TypeError, match="Invalid data type"):
        test_poly.fit(x_type, y_type)

    with pytest.raises(TypeError, match="must be array-like"):
        test_poly.fit(x_type2, y_type2)

    with pytest.raises(ValueError, match="Expected a 1-D array"):
        test_poly.fit(array_2d, y)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        test_poly.fit(x, array_mismatch)

def test_poly_predict_before_fit():
    test_poly = polynomial_regression(degree=2)
    x = np.array([1.0, 2.0])

    with pytest.raises(RuntimeError, match="not fitted yet"):
        test_poly.predict(x)
