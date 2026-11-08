import pytest
import numpy as np
from ml_library.models.regression.linear_regression import _LinearRegression, OrdinaryLeastSquares, Ridge, Lasso

def test_yet_not_fitted_error_message():
    X = np.array([[1,-1],[2,5],[0,-2],[-3,2],[3,4]])
    model = _LinearRegression()

    with pytest.raises(RuntimeError) as excep1:
        model.predict(X)
    assert str(excep1.value) == "Model is not yet fitted, call .fit() first."

def test_ols_shape():
    X = np.array([[1,-1],[2,5],[0,-2],[-3,2],[3,4]])
    y = np.array([1,2,3,4,5])
    model = OrdinaryLeastSquares()

    model.fit(X,y)
    assert len(model.coef) == 2
    assert isinstance(model.intercept, float)

    model.fit(X,y,intercept=False)
    assert len(model.coef) == 2
    assert isinstance(model.intercept, float)
    assert model.intercept == 0

def test_ridge_shape():
    X = np.array([[1,-1],[2,5],[0,-2],[-3,2],[3,4]])
    y = np.array([1,2,3,4,5])
    model = Ridge()

    model.fit(X,y, lam=1)
    assert len(model.coef) == 2
    assert isinstance(model.intercept, float)

    model.fit(X,y, lam=1,intercept=False)
    assert len(model.coef) == 2
    assert isinstance(model.intercept, float)
    assert model.intercept == 0

def test_lasso_shape():
    X = np.array([[1,-1],[2,5],[0,-2],[-3,2],[3,4]])
    y = np.array([1,2,3,4,5])
    model = Lasso()

    model.fit(X,y,lam=1)
    assert len(model.coef) == 2
    assert isinstance(model.intercept, float)

    model.fit(X,y,lam=1,intercept=False)
    assert len(model.coef) == 2
    assert isinstance(model.intercept, float)
    assert model.intercept == 0

def test_ols_constant_value_regression():
    X = np.array([[1,2,3],[0,2,9],[-1,8,6],[8,5,3],[9,4,-3],[30,40,50],[-1,-2,-3],[0,-1,2],[11,-10,9],[5,3,1]])
    y = np.array([1,1,1,1,1,1,1,1,1,1])
    model = OrdinaryLeastSquares()

    model.fit(X,y)

    assert (np.allclose(model.coef, 0)) and (model.intercept == pytest.approx(1))

def test_ridge_constant_value_regression():
    X = np.array([[1,2,3],[0,2,9],[-1,8,6],[8,5,3],[9,4,-3],[30,40,50],[-1,-2,-3],[0,-1,2],[11,-10,9],[5,3,1]])
    y = np.array([1,1,1,1,1,1,1,1,1,1])
    model = Ridge()
    model.fit(X,y, lam=1)

    assert (np.allclose(model.coef, 0)) and (model.intercept == pytest.approx(1))

def test_lasso_constant_value_regression():
    X = np.array([[1,2,3],[0,2,9],[-1,8,6],[8,5,3],[9,4,-3],[30,40,50],[-1,-2,-3],[0,-1,2],[11,-10,9],[5,3,1]])
    y = np.array([1,1,1,1,1,1,1,1,1,1])
    model = Lasso()
    model.fit(X,y, lam=1)

    assert (np.allclose(model.coef, 0)) and (model.intercept == pytest.approx(1))

def test_ols_perfect_plane_regression():
    X = np.array([[1,2,3,4],[0,2,9,1],[-1,8,6,2],[8,5,3,9],[9,4,-3,1],[30,40,50,60],[-1,-2,-3,-8],[0,-1,2,4],[11,-10,9,-3],[5,3,1,-6]])
    y = 1*X[:,0] + 2*X[:,1] + 3*X[:,2] + 4*X[:,3] + 5
    model = OrdinaryLeastSquares()

    model.fit(X,y)

    assert (model.coef == pytest.approx([1,2,3,4])) and (model.intercept == pytest.approx(5))

def test_ridge_perfect_plane_regression():
    X = np.array([[1,2,3,4],[0,2,9,1],[-1,8,6,2],[8,5,3,9],[9,4,-3,1],[30,40,50,60],[-1,-2,-3,-8],[0,-1,2,4],[11,-10,9,-3],[5,3,1,-6]])
    y = 1*X[:,0] + 2*X[:,1] + 3*X[:,2] + 4*X[:,3] + 5
    model = Ridge()

    model.fit(X,y,lam=1e-10)

    assert (model.coef == pytest.approx([1,2,3,4])) and (model.intercept == pytest.approx(5))

def test_lasso_perfect_plane_regression():
    X = np.array([[1,2,3,4],[0,2,9,1],[-1,8,6,2],[8,5,3,9],[9,4,-3,1],[30,40,50,60],[-1,-2,-3,-8],[0,-1,2,4],[11,-10,9,-3],[5,3,1,-6]])
    y = 1*X[:,0] + 2*X[:,1] + 3*X[:,2] + 4*X[:,3] + 5
    model = Lasso()

    model.fit(X,y,lam=1e-10)

    assert (model.coef == pytest.approx([1,2,3,4])) and (model.intercept == pytest.approx(5))

def test_ridge_normalize_test():
    X = [[1,2,3,4],[1,-2,3,5],[1,0,2,6],[1,1,1,1]]
    y = [1,2,3,4]
    X_n, y_n = Ridge()._normalize(X,y)

    np.testing.assert_allclose(X_n, [[0., 1.18321596, 0.90453403, 0.],[0., -1.52127766, 0.90453403, 0.53452248],[0., -0.16903085, -0.30151134, 1.06904497],[ 0., 0.50709255, -1.50755672, -1.60356745]])
    assert y_n == pytest.approx([-1.34164079,-0.4472136, 0.4472136,1.34164079])

def test_lasso_normalize_test():
    X = [[1,2,3,4],[1,-2,3,5],[1,0,2,6],[1,1,1,1]]
    y = [1,2,3,4]
    X_n, y_n = Lasso()._normalize(X,y)

    np.testing.assert_allclose(X_n, [[0., 1.18321596, 0.90453403, 0.],[0., -1.52127766, 0.90453403, 0.53452248],[0., -0.16903085, -0.30151134, 1.06904497],[ 0., 0.50709255, -1.50755672, -1.60356745]])
    assert y_n == pytest.approx([-1.34164079,-0.4472136, 0.4472136,1.34164079])

def test_ols_shape_error_message():
    X = np.array([[1,2,3],[0,2,9],[-1,8,6],[8,5,3],[9,4,-3],[30,40,50],[-1,-2,-3],[0,-1,2],[11,-10,9],[5,3,1]])
    X_1d = np.array([-11,32,93,14,25,36])
    y = np.array([1,2,3,4,5,6])
    y_2d = np.array([[1,2],[2,3],[4,5]])

    model = OrdinaryLeastSquares()

    with pytest.raises(ValueError) as excep1:
        model.fit(X,y)
    assert str(excep1.value) == "Dimension mismatch: The number of rows in feature matrix (X) must equal the number of samples in the target vector (y)."

    with pytest.raises(ValueError) as excep2:
        model.fit(X,y, intercept=False)
    assert str(excep2.value) == "Dimension mismatch: The number of rows in feature matrix (X) must equal the number of samples in the target vector (y)."

    with pytest.raises(ValueError) as excep3:
        model.fit(X_1d, y)
    assert str(excep3.value) == "Expected a 2-D array of shape (n_points, n_features)."

    with pytest.raises(ValueError) as excep4:
        model.fit(X_1d, y, intercept=False)
    assert str(excep4.value) == "Expected a 2-D array of shape (n_points, n_features)."

    with pytest.raises(ValueError) as excep5:
        model.fit(X, y_2d)
    assert str(excep5.value) == "Expected a 1-D array of shape (n_points, )."

    with pytest.raises(ValueError) as excep6:
        model.fit(X, y_2d, intercept=False)
    assert str(excep6.value) == "Expected a 1-D array of shape (n_points, )."

def test_ols_type_error_message():
    X = np.array([[1,2],[-1,1],[3,2]])
    X_str = np.array([['a', 'b'], ['c', 'd']])
    y = np.array([1,2])
    y_str = np.array(["e","f"])
    intercept = 'g'

    model = OrdinaryLeastSquares()

    with pytest.raises(TypeError) as excep1:
        model.fit(X_str, y)
    assert str(excep1.value) == f"Invalid data type: Expected feature matrix (X) to have a numeric data type, got '{X_str.dtype}'."

    with pytest.raises(TypeError) as excep2:
        model.fit(X_str, y, intercept=False)
    assert str(excep2.value) == f"Invalid data type: Expected feature matrix (X) to have a numeric data type, got '{X_str.dtype}'."

    with pytest.raises(TypeError) as excep3:
        model.fit(X, y_str)
    assert str(excep3.value) == f"Invalid data type: Expected target vector (y) to have a numeric data type, got '{y_str.dtype}'."

    with pytest.raises(TypeError) as excep4:
        model.fit(X, y_str, intercept=False)
    assert str(excep4.value) == f"Invalid data type: Expected target vector (y) to have a numeric data type, got '{y_str.dtype}'."

    with pytest.raises(TypeError) as excep5:
        model.fit(X, y, intercept=intercept)
    assert str(excep5.value) == f"Invalid data type: Expected intercept flag (intercept) to have a bool data type, got '{type(intercept).__name__}'."

def test_ols_non_invertible_matrix_error_message():
    X = np.array([[1,0,1],[0,1,1],[1,1,2]])
    X_with_intercept = np.array([[1,0,0],[0,1,0],[0,0,1]])
    y = np.array([1,2,3])
    model = OrdinaryLeastSquares()

    with pytest.raises(ValueError) as excep1:
        model.fit(X_with_intercept, y, intercept=True)
    assert str(excep1.value) == "Normal equations not solvable: The design matrix must have full column rank."

    with pytest.raises(ValueError) as excep2:
        model.fit(X, y, intercept=False)
    assert str(excep2.value) == "Normal equations not solvable: The design matrix must have full column rank."

def test_ridge_shape_error_message():
    X = np.array([[1,2,3],[0,2,9],[-1,8,6],[8,5,3],[9,4,-3],[30,40,50],[-1,-2,-3],[0,-1,2],[11,-10,9],[5,3,1]])
    X_1d = np.array([-11,32,93,14,25,36])
    y = np.array([1,2,3,4,5,6])
    y_2d = np.array([[1,2],[2,3],[4,5]])

    model = Ridge()

    with pytest.raises(ValueError) as excep1:
        model.fit(X,y, lam=1)
    assert str(excep1.value) == "Dimension mismatch: The number of rows in feature matrix (X) must equal the number of samples in the target vector (y)."

    with pytest.raises(ValueError) as excep2:
        model.fit(X,y, lam=1, intercept=False)
    assert str(excep2.value) == "Dimension mismatch: The number of rows in feature matrix (X) must equal the number of samples in the target vector (y)."

    with pytest.raises(ValueError) as excep3:
        model.fit(X_1d, y, lam=1)
    assert str(excep3.value) == "Expected a 2-D array of shape (n_points, n_features)."

    with pytest.raises(ValueError) as excep4:
        model.fit(X_1d, y, lam=1, intercept=False)
    assert str(excep4.value) == "Expected a 2-D array of shape (n_points, n_features)."

    with pytest.raises(ValueError) as excep5:
        model.fit(X, y_2d, lam=1)
    assert str(excep5.value) == "Expected a 1-D array of shape (n_points, )."

    with pytest.raises(ValueError) as excep6:
        model.fit(X, y_2d, lam=1, intercept=False)
    assert str(excep6.value) == "Expected a 1-D array of shape (n_points, )."

def test_ridge_type_error_message():
    X = np.array([[1,2],[-1,1],[3,2]])
    X_str = np.array([['a', 'b'], ['c', 'd']])
    y = np.array([1,2])
    y_str = np.array(["e","f"])
    intercept = 'g'
    lam = 'h'

    model = Ridge()

    with pytest.raises(TypeError) as excep1:
        model.fit(X_str, y, lam=1)
    assert str(excep1.value) == f"Invalid data type: Expected feature matrix (X) to have a numeric data type, got '{X_str.dtype}'."

    with pytest.raises(TypeError) as excep2:
        model.fit(X_str, y, lam=1, intercept=False)
    assert str(excep2.value) == f"Invalid data type: Expected feature matrix (X) to have a numeric data type, got '{X_str.dtype}'."

    with pytest.raises(TypeError) as excep3:
        model.fit(X, y_str, lam=1)
    assert str(excep3.value) == f"Invalid data type: Expected target vector (y) to have a numeric data type, got '{y_str.dtype}'."

    with pytest.raises(TypeError) as excep4:
        model.fit(X, y_str, lam=1, intercept=False)
    assert str(excep4.value) == f"Invalid data type: Expected target vector (y) to have a numeric data type, got '{y_str.dtype}'."

    with pytest.raises(TypeError) as excep5:
        model.fit(X, y, lam=1, intercept=intercept)
    assert str(excep5.value) == f"Invalid data type: Expected intercept flag (intercept) to have a bool data type, got '{type(intercept).__name__}'."

    with pytest.raises(TypeError) as excep6:
        model.fit(X, y, lam=lam)
    assert str(excep6.value) == f"Invalid data type: Expected lambda hyperparameter (lam) to have a float data type, got '{type(lam).__name__}'."

    with pytest.raises(TypeError) as excep7:
        model.fit(X, y, lam=lam, intercept=False)
    assert str(excep7.value) == f"Invalid data type: Expected lambda hyperparameter (lam) to have a float data type, got '{type(lam).__name__}'."

    with pytest.raises(ValueError) as excep8:
        model.fit(X, y, lam=-1)
    assert str(excep8.value) == "Invalid value for (lam): Lambda hyperparameter (lam) must be a non-negative value."

    with pytest.raises(ValueError) as excep9:
        model.fit(X, y, lam=-1, intercept=False)
    assert str(excep9.value) == "Invalid value for (lam): Lambda hyperparameter (lam) must be a non-negative value."

def test_lasso_shape_error_message():
    X = np.array([[1,2,3],[0,2,9],[-1,8,6],[8,5,3],[9,4,-3],[30,40,50],[-1,-2,-3],[0,-1,2],[11,-10,9],[5,3,1]])
    X_1d = np.array([-11,32,93,14,25,36])
    y = np.array([1,2,3,4,5,6])
    y_2d = np.array([[1,2],[2,3],[4,5]])

    model = Lasso()

    with pytest.raises(ValueError) as excep1:
        model.fit(X,y, lam=1)
    assert str(excep1.value) == "Dimension mismatch: The number of rows in feature matrix (X) must equal the number of samples in the target vector (y)."

    with pytest.raises(ValueError) as excep2:
        model.fit(X,y, lam=1, intercept=False)
    assert str(excep2.value) == "Dimension mismatch: The number of rows in feature matrix (X) must equal the number of samples in the target vector (y)."

    with pytest.raises(ValueError) as excep3:
        model.fit(X_1d, y, lam=1)
    assert str(excep3.value) == "Expected a 2-D array of shape (n_points, n_features)."

    with pytest.raises(ValueError) as excep4:
        model.fit(X_1d, y, lam=1, intercept=False)
    assert str(excep4.value) == "Expected a 2-D array of shape (n_points, n_features)."

    with pytest.raises(ValueError) as excep5:
        model.fit(X, y_2d, lam=1)
    assert str(excep5.value) == "Expected a 1-D array of shape (n_points, )."

    with pytest.raises(ValueError) as excep6:
        model.fit(X, y_2d, lam=1, intercept=False)
    assert str(excep6.value) == "Expected a 1-D array of shape (n_points, )."

def test_lasso_type_error_message():
    X = np.array([[1,2],[-1,1],[3,2]])
    X_str = np.array([['a', 'b'], ['c', 'd']])
    y = np.array([1,2])
    y_str = np.array(["e","f"])
    intercept = 'g'
    lam = 'h'
    eps = 'i'
    max_iter = 2.5

    model = Lasso()

    with pytest.raises(TypeError) as excep1:
        model.fit(X_str, y, lam=1)
    assert str(excep1.value) == f"Invalid data type: Expected feature matrix (X) to have a numeric data type, got '{X_str.dtype}'."

    with pytest.raises(TypeError) as excep2:
        model.fit(X_str, y, lam=1, intercept=False)
    assert str(excep2.value) == f"Invalid data type: Expected feature matrix (X) to have a numeric data type, got '{X_str.dtype}'."

    with pytest.raises(TypeError) as excep3:
        model.fit(X, y_str, lam=1)
    assert str(excep3.value) == f"Invalid data type: Expected target vector (y) to have a numeric data type, got '{y_str.dtype}'."

    with pytest.raises(TypeError) as excep4:
        model.fit(X, y_str, lam=1, intercept=False)
    assert str(excep4.value) == f"Invalid data type: Expected target vector (y) to have a numeric data type, got '{y_str.dtype}'."

    with pytest.raises(TypeError) as excep5:
        model.fit(X, y, lam=1, intercept=intercept)
    assert str(excep5.value) == f"Invalid data type: Expected intercept flag (intercept) to have a bool data type, got '{type(intercept).__name__}'."

    with pytest.raises(TypeError) as excep6:
        model.fit(X, y, lam=lam)
    assert str(excep6.value) == f"Invalid data type: Expected lambda hyperparameter (lam) to have a float data type, got '{type(lam).__name__}'."

    with pytest.raises(TypeError) as excep7:
        model.fit(X, y, lam=lam, intercept=False)
    assert str(excep7.value) == f"Invalid data type: Expected lambda hyperparameter (lam) to have a float data type, got '{type(lam).__name__}'."

    with pytest.raises(ValueError) as excep8:
        model.fit(X, y, lam=-1)
    assert str(excep8.value) == "Invalid value for (lam): Lambda hyperparameter (lam) must be a non-negative value."

    with pytest.raises(ValueError) as excep9:
        model.fit(X, y, lam=-1, intercept=False)
    assert str(excep9.value) == "Invalid value for (lam): Lambda hyperparameter (lam) must be a non-negative value."

    with pytest.raises(TypeError) as excep10:
        model.fit(X, y, lam=1, intercept=True, eps=eps)
    assert str(excep10.value) == f"Invalid data type: Expected epsilon convergence parameter (eps) to have a float data type, got '{type(eps).__name__}'."

    with pytest.raises(TypeError) as excep11:
        model.fit(X, y, lam=1, intercept=False, eps=eps)
    assert str(excep11.value) == f"Invalid data type: Expected epsilon convergence parameter (eps) to have a float data type, got '{type(eps).__name__}'."

    with pytest.raises(ValueError) as excep12:
        model.fit(X, y, lam=1, intercept=True, eps=-1)
    assert str(excep12.value) == "Invalid value for (eps): Epsilon convergence parameter (eps) must be a non-negative value."

    with pytest.raises(ValueError) as excep13:
        model.fit(X, y, lam=1, intercept=False, eps=-1)
    assert str(excep13.value) == "Invalid value for (eps): Epsilon convergence parameter (eps) must be a non-negative value."

    with pytest.raises(TypeError) as excep14:
        model.fit(X, y, lam=1, intercept=True, eps=1, max_iter=max_iter)
    assert str(excep14.value) == f"Invalid data type: Expected maximum of iterations convergence parameter (max_iter) to have a int data type, got '{type(max_iter).__name__}'."

    with pytest.raises(TypeError) as excep15:
        model.fit(X, y, lam=1, intercept=False, eps=1, max_iter=max_iter)
    assert str(excep15.value) == f"Invalid data type: Expected maximum of iterations convergence parameter (max_iter) to have a int data type, got '{type(max_iter).__name__}'."

    with pytest.raises(ValueError) as excep16:
        model.fit(X, y, lam=1, intercept=True, eps=1, max_iter=0)
    assert str(excep16.value) == "Invalid value for (max_iter): Maximum of iterations convergence parameter (max_iter) must be a positive value."

    with pytest.raises(ValueError) as excep17:
        model.fit(X, y, lam=1, intercept=False, eps=1, max_iter=0)
    assert str(excep17.value) == "Invalid value for (max_iter): Maximum of iterations convergence parameter (max_iter) must be a positive value."
