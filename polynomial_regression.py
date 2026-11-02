import numpy as np
from linear_regression import ordinary_least_squares, ridge, lasso

class polynomial_regression:
    """
    A public class for polynomial regression using OLS, Ridge, or Lasso.

    Args:
        degree (int): Degree of the polynomial used for regression.
        model_choice (str, optional): Method to be used for regression. Options are 'ols' (default), 'ridge', and 'lasso'.
        intercept (bool, optional): Whether the regression has an intercept. Default is True.
        lam (float, optional): Lambda hyperparameter for 'ridge' and 'lasso' methods. Default is 1.0.
        eps (float, optional): Tolerance for 'lasso' method. Default is 1e-4.
        max_iter (int, optional): Maximum number of iterations for 'lasso' method. Default is 10000.
    
    Attributes:
        degree (int): Degree of the polynomial used for regression.
        has_intercept (bool): Whether the regression has an intercept.
        model_choice (str): Model chosen for the regression.
        model (object): Model as an object from inherited from the linear regression.
        coef (np.ndarray): Calculated coefficients for the regression.
        model_intercept (float): Calculated intercept for the regression.
        lam (float): Regularization hyperparameter.
        eps (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
    """
    def __init__(self, degree: int, model_choice: str = 'ols', intercept=True, lam: float = 1.0, eps: float = 1e-4, max_iter: int = 10000) -> None:
        self._exceptions_init(degree, model_choice, intercept, lam, eps, max_iter)

        self.degree = degree
        self.has_intercept = intercept
        self.model_choice = model_choice
        self.coef = None

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
        """
        Private method to verify exceptions in the initialization.

        Args:
            degree (int): Degree of the polynomial regression.
            model_choice (str): Model chosen for the regression.
            intercept (bool): Whether the regression has an intercept.
            lam (float): Regularization hyperparameter.
            eps (float): Convergence tolerance.
            max_iter (int): Maximum number of iterations.
        """
        if not isinstance(degree, int):
            raise TypeError(f"Invalid data type: Expected polynomial degree for regression (degree) to have an int data type, got '{type(degree).__name__}'")
        
        if degree <= 0:
            raise ValueError("Invalid value for (degree): Polynomial degree for regression (degree) must be a positive value.")
        
        if not isinstance(model_choice, str) or model_choice not in ['ols', 'ridge', 'lasso']:
            raise ValueError("Invalid value for (model_choice): Model choice for polynomial regression (model_choice) must be 'ols', 'ridge' or 'lasso'.")

        if not isinstance(intercept, bool):
            raise TypeError(f"Invalid data type: Expected intercept flag (intercept) to have a bool data type, got '{type(intercept).__name__}'.")

        if not isinstance(lam, (float, int)):
            raise TypeError(f"Invalid data type: Expected lambda hyperparameter (lam) to have a float or int data type, got '{type(lam).__name__}'.")

        if lam <= 0:
            raise ValueError("Invalid value for (lam): Lambda hyperparameter (lam) must be a positive value.")

        if not isinstance(eps, (float, int)):
            raise TypeError(f"Invalid data type: Expected epsilon convergence parameter (eps) to have a float or int data type, got '{type(eps).__name__}'.")

        if eps <= 0:
            raise ValueError("Invalid value for (eps): Epsilon convergence parameter (eps) must be a positive value.")

        if not isinstance(max_iter, int):
            raise TypeError(f"Invalid data type: Expected maximum of iterations convergence parameter (max_iter) to have a int data type, got '{type(max_iter).__name__}'.")
        
        if max_iter <= 0:
            raise ValueError("Invalid value for (max_iter): Maximum of iterations convergence parameter (max_iter) must be a positive value.")

    def _exceptions_data(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """
        Private method to verify exceptions in inputs.

        Args:
            x (np.ndarray): Feature vector in the input data.
            y (np.ndarray): Target vector in the input data.

        Returns:
            np.ndarray
        """
        if not isinstance(x, (list, np.ndarray, tuple)):
            raise TypeError(f"Invalid data type: Feature vector must be array-like (list, numpy array or tuple), got '{type(x).__name__}'.")

        try:
            x_array = np.asarray(x, dtype=np.float64)
        except (ValueError, TypeError):
            raise TypeError("Invalid data type: Feature vector must contain only numerical values.")

        if x_array.ndim != 1:
            raise ValueError(f"Expected a 1-D array of shape (n_points) for polynomial regression input, got '{x_array.ndim}-D'.")

        if y is not None:
            if not isinstance(y, (list, np.ndarray, tuple)):
                raise TypeError(f"Invalid data type: Target vector must be array-like (list, numpy array or tuple), got '{type(y).__name__}'.")

            try:
                y_array = np.asarray(y, dtype=np.float64)
            except (ValueError, TypeError):
                raise TypeError("Invalid data type: Target vector must contain only numerical values.")
            
            if y_array.ndim != 1:
                raise ValueError(f"Expected a 1-D array of shape (n_points) for polynomial regression output, got '{y_array.ndim}-D'.")
            
            if x_array.shape[0] != y_array.shape[0]:
                raise ValueError("Dimension mismatch: Feature vector (x) and target vector (y) must have the same length.")

            return x_array, y_array
            
        return x_array

    def _build_design(self, x: np.ndarray) -> np.ndarray:
        """
        Private method to build the design matrix.

        Args:
            x (np.ndarray): Feature vector used for the regression.
        
        Returns:
            np.ndarray
        """
        X = np.array([[val_xi**d for d in range(1, self.degree + 1)] for val_xi in x])
        return X
        
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Obtains the polynomial regression using the method defined in the initialization.

        Args:
            x (np.ndarray): Feature vector used for the regression.
            y (np.ndarray): Target vector used for the regression.
        """
        x_array, y_array = self._exceptions_data(x, y)
        X = self._build_design(x_array)

        if self.model_choice == 'ols':
            self.model.fit(X, y_array, intercept=self.has_intercept)
        elif self.model_choice == 'ridge':
            self.model.fit(X, y_array, intercept=self.has_intercept, lam=self.lam)
        else:
            self.model.fit(X, y_array, intercept=self.has_intercept, lam=self.lam, eps=self.eps, max_iter=self.max_iter)

        self.coef = self.model.coef
        self.model_intercept = self.model.intercept

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the predicted values for a vector of inputs, using the previously trained model.

        Args:
            x (np.ndarray): Input values for the prediction.
        
        Returns:
            np.ndarray
        """
        if self.coef == None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

        x_array = self._exceptions_data(x)
        X = self._build_design(x_array)

        preds = self.model.predict(X)
        return preds

    @property
    def coefficients(self):
        """
        Returns a list of coefficients, including the intercept.

        Returns:
            np.ndarray
        """
        return np.concatenate((self.model_intercept, self.coef), axis=None)