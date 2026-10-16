import numpy as np
from linear_regression import ordinary_least_squares, ridge, lasso

# TODO: docstrings
class polynomial_regression:
    """
    A public class for polynomial regression using OLS, Ridge, or Lasso.
    """
    def __init__(self, degree: int, model_choice: str = 'ols', intercept=True, lam: float = 1.0, eps: float = 1e-4, max_iter: int = 10000) -> None:
        self._exceptions_init(degree, model_choice, intercept, lam, eps, max_iter)

        self.degree = degree
        self.has_intercept = intercept
        self.model_choice = model_choice

        self.lam = lam
        self.eps = eps
        self.max_iter = max_iter

        if model_choice == 'ols':
            self.model = ordinary_least_squares()
        elif model_choice == 'ridge':
            self.model = ridge()
        elif model_choice == 'lasso':
            self.model = lasso()
        
    def _exceptions_init(self, degree: int, model_choice: str, intercept: bool, lam: float, eps: float, max_iter: int) -> None:
        if not isinstance(degree, int):
            raise TypeError(f"Invalid data type: Expected polynomial degree for regression (degree) to have an int data type, got '{type(degree).__name__}'")
        
        if degree <= 0:
            raise ValueError("Invalid value for (degree): Polynomial degree for regression (degree) must be a positive value.")
        
        if not isinstance(model_choice, str) or model_choice not in ['ols', 'ridge', 'lasso']:
            raise ValueError("Invalid value for (model_choice): Model choice for polynomial regression (model_choice) must be 'ols', 'ridge' or 'lasso'.")

        if not isinstance(intercept, bool):
            raise TypeError(f"Invalid data type: Expected intercept flag (intercept) to have a bool data type, got '{type(intercept).__name__}'.")

        if not isinstance(lam, float):
            raise TypeError(f"Invalid data type: Expected lambda hyperparameter (lam) to have a float data type, got '{type(lam).__name__}'.")

        if lam <= 0:
            raise ValueError("Invalid value for (lam): Lambda hyperparameter (lam) must be a positive value.")

        if not isinstance(eps, float):
            raise TypeError(f"Invalid data type: Expected epsilon convergence parameter (eps) to have a float data type, got '{type(eps).__name__}'.")

        if eps <= 0:
            raise ValueError("Invalid value for (eps): Epsilon convergence parameter (eps) must be a positive value.")

        if not isinstance(max_iter, int):
            raise TypeError(f"Invalid data type: Expected maximum of iterations convergence parameter (max_iter) to have a int data type, got '{type(max_iter).__name__}'.")
        
        if max_iter <= 0:
            raise ValueError("Invalid value for (max_iter): Maximum of iterations convergence parameter (max_iter) must be a positive value.")

    def _exceptions_data(self, x: np.ndarray, y: np.ndarray = None) -> None:
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError(f"Invalid data type: Expected feature vector (x) to have a numeric data type, got '{x.dtype}'.")

        if x.ndim != 1:
            raise ValueError(f"Expected a 1-D array of shape (n_points) for polynomial regression input, got '{x.ndim}-D'.")

        if y is not None:
            if not np.issubdtype(y.dtype, np.number):
                raise TypeError(f"Invalid data type: Expected target vector (y) to have a numeric data type, got '{y.dtype}'.")
            
            if y.ndim != 1:
                raise ValueError(f"Expected a 1-D array of shape (n_points) for polynomial regression output, got '{y.ndim}-D'.")
            
            if x.shape[0] != y.shape[0]:
                raise ValueError("Dimension mismatch: Feature vector (x) and target vector (y) must have the same length.")

    def _build_design(self, x: np.ndarray) -> np.ndarray:
        X = np.array([[val_xi**d for d in range(1, self.degree + 1)] for val_xi in x])
        return X
        
    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        self._exceptions_data(x, y)
        X = self._build_design(x)

        if self.model_choice == 'ols':
            self.model.train(X, y, intercept=self.has_intercept)
        elif self.model_choice == 'ridge':
            self.model.train(X, y, intercept=self.has_intercept, lam=self.lam)
        else:
            self.model.train(X, y, intercept=self.has_intercept, lam=self.lam, eps=self.eps, max_iter=self.max_iter)

        self.coef = self.model.coef
        self.model_intercept = self.model.intercept

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._exceptions_data(x)
        X = self._build_design(x)

        preds = self.model.predict(X)
        return preds

# TODO: MAE, R2_adjusted. Adapt implementation of other metrics

def RSS(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum((y - y_pred)**2)

def MSE(y: np.ndarray, y_pred: np.ndarray) -> float:
    return RSS(y, y_pred)/len(y)

def RMSE(y: np.ndarray, y_pred: np.ndarray) -> float:
    pass

def R2(y: np.ndarray, y_pred: np.ndarray) -> float:
    pass

if __name__ == "__main__":
    # TODO: remove temporary tests. 
    # TODO: consider numerical stability (i.e.: deg=5 for x_test with 7 entries below).
    # TODO: alert for insufficient data for degree of regression (i.e.: deg=5 below).
    x_test = np.array([-2, -1, 0, 1, 2])
    y_test = 3 + 2*x_test + x_test**2
    np.random.seed(67)
    error = np.random.normal(0.0, 0.5, size=len(x_test))
    y_noise = y_test + error
    deg = 4

    x_test2 = np.array([[-3, -2, -1, 0, 1, 2, 3], [-30, -20, -10, 0, 10, 20, 30]])
    print(x_test2.ndim)

    # Test OLS
    test_ols = polynomial_regression(degree=deg, model_choice='ols', intercept=True)
    test_ols.train(x_test, y_noise)
    print(f"OLS: {test_ols.model_intercept:.3f} and {test_ols.coef}")

    # Test Ridge
    test_ridge = polynomial_regression(degree=deg, model_choice='ridge', intercept=True, lam=0.1)
    test_ridge.train(x_test, y_noise)
    print(f"Ridge: {test_ridge.model_intercept:.3f} and {test_ridge.coef}")

    # Test Lasso
    test_lasso = polynomial_regression(degree=deg, model_choice='lasso', intercept=True, lam=0.1)
    test_lasso.train(x_test, y_noise)
    print(f"Lasso: {test_lasso.model_intercept:.3f} and {test_lasso.coef}")

    # Test exceptions
    try:
        test_ols.train(x_test2, y_noise)
    except ValueError as V:
        print(f"Exception: {V}")