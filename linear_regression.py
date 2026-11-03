import numpy as np

class _linear_regression:
    """A private base class for linear regression.

    Add the predict method and some metrics for linear regression.

    Attributes:
        coef (np.ndarray): The array with the slope coefficient of each variable.
        intercept (float, optional): The value of the intercept coefficient.
    """
    def __init__(self):
        self._coef = []
        self._intercept = 0.0
        return

    @property
    def coef(self):
        """Provide access to the private attribute _coef.

        Returns:
            np.ndarray: The regression coefficients.
        """
        return self._coef

    @property
    def intercept(self):
        """Provide access to the private attribute _intercept.

        Returns:
            float: The regression intercept coefficient.
        """
        return self._intercept
    
    def fit(self):
        """Abstract method to fit the model"""
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the target values given the feature variables.

        Evaluate the linear function on the X feature matrix.

        Args:
            X (np.ndarray): The feature matrix.

        Returns:
            np.ndarray: The predicted value for each line of the X matrix.
        """
        X_p = np.array(X)

        self._exceptions(X_p, np.zeros(X_p.shape[0]))
        
        y = X_p @ self._coef + self._intercept
        return y

    def RSS(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate the residual sum of squares metric (RSS).

        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The real values of the target variable.

        Returns:
            float: The residual sum of squares.
        """
        y_r = np.array(y)
        X_r = np.array(X)

        self._exceptions(X_r, y_r)
        
        y_hat = self.predict(X_r)
        rss = np.sum((y_r-y_hat)**2)
        return rss

    def MSE(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate the mean squared error metric (MSE).
        
        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The real values of the target variable.

        Returns:
            float: The mean squared error.
        """
        return self.RSS(X,y)/len(y)

    def RMSE(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate the root mean square error metric (RMSE).
        
        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The real values of the target variable.

        Returns:
            float: The root mean square error.
        """
        return np.sqrt(self.MSE(X,y))

    def R2(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate the R-squared metric (R^2).
        
        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The real values of the target variable.

        Returns:
            float: The R-squared.
        """
        X_r = np.array(X)
        y_r = np.array(y)

        self._exceptions(X_r, y_r)

        y_hat = self.predict(X_r)

        r2 = 1 - (np.sum((y_r-y_hat)**2))/(np.sum((y_r-np.mean(y_r))**2))
        return r2

    def _exceptions(self, X: np.ndarray, y: np.ndarray) -> None:
        """Private method to log internal exceptions"""
        if(self._coef==[]):
            raise ValueError("Model is not yet fitted, call .fit() first.")
        
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"Invalid data type: Expected feature matrix (X) to have a numeric data type, got '{X.dtype}'.")
        
        if not np.issubdtype(y.dtype, np.number):
            raise TypeError(f"Invalid data type: Expected target vector (y) to have a numeric data type, got '{y.dtype}'.")
        
        if(X.ndim!=2):
            raise ValueError("Expected a 2-D array of shape (n_points, n_features).")
        
        if(y.ndim!=1):
            raise ValueError("Expected a 1-D array of shape (n_points, ).")
        
        if(X.shape[0]!=y.shape[0]):
            raise ValueError("Dimension mismatch: The number of rows in feature matrix (X) must equal the number of samples in the target vector (y).")
        
        if(X.shape[1]!=len(self._coef)):
            raise ValueError("Dimension mismatch: The number of columns in feature matrix (X) must equal the number of model coefficients.")
        return

class ordinary_least_squares(_linear_regression):
    """A public class for ordinary least squares linear regression.

    This class inherits the private class _linear_regression and add the fit method for ordinary least squares regression.
    """
    def fit(self, X: np.ndarray, y: np.ndarray, intercept: bool = True) -> None:
        """Fit the ordinary least squares model.

        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The empirical target values.
            intercept (bool, optional): Decide whether to fit the model with an intercept coefficient or not.

        Returns:
            None

        Attributes Updated:
            coef (np.ndarray): Calculated coefficients for linear regression.
            intercept (float, optional): Calculated intercept coefficient for linear regression.
        """
        X_t = np.array(X)
        y_t = np.array(y)

        self._ols_exceptions(X_t, y_t, intercept)

        self._intercept = 0.0

        if(intercept):
            ones_column = np.ones((X_t.shape[0],1))
            X_t = np.hstack((ones_column, X_t))

        beta = np.linalg.solve(X_t.T @ X_t, X_t.T @ y_t)

        if(intercept):
            self._intercept = beta[0]
            beta = beta[1:]

        self._coef = beta
        return

    def _ols_exceptions(self, X: np.ndarray, y: np.ndarray, intercept: bool) -> None:
        """Private method to log internal exceptions"""
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"Invalid data type: Expected feature matrix (X) to have a numeric data type, got '{X.dtype}'.")
        
        if not np.issubdtype(y.dtype, np.number):
            raise TypeError(f"Invalid data type: Expected target vector (y) to have a numeric data type, got '{y.dtype}'.")

        if not isinstance(intercept, bool):
            raise TypeError(f"Invalid data type: Expected intercept flag (intercept) to have a bool data type, got '{type(intercept).__name__}'.")
        
        if(X.ndim!=2):
            raise ValueError("Expected a 2-D array of shape (n_points, n_features).")
        
        if(y.ndim!=1):
            raise ValueError("Expected a 1-D array of shape (n_points, ).")
        
        if(X.shape[0]!=y.shape[0]):
            raise ValueError("Dimension mismatch: The number of rows in feature matrix (X) must equal the number of samples in the target vector (y).")
        
        X_aux = X
        if(intercept):
            ones_column = np.ones((X_aux.shape[0],1))
            X_aux = np.hstack((ones_column, X_aux))
        
        if(np.linalg.matrix_rank(X_aux)!=X_aux.shape[1]):
            raise ValueError("Normal equations not solvable: The design matrix must have full column rank.")
        return

class ridge(_linear_regression):
    """A public class for ridge linear regression.

    This class inherits the private class _linear_regression and add the fit method for ridge regression.
    """
    def fit(self, X: np.ndarray, y: np.ndarray, lam: float, intercept: bool = True) -> None:
        """Fit the ridge model.

        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The empirical target values.
            lam (float): Regularization ridge hyperparameter. Should be a positive value. Higher values force the slope coefficients closer to 0.
            intercept (bool, optional): Decide whether to fit the model with an intercept coefficient or not.

        Returns:
            None

        Attributes Updated:
            coef (np.ndarray): Calculated coefficients for linear regression.
            intercept (float, optional): Calculated intercept coefficient for linear regression.
        """
        X_t = np.array(X)
        y_t = np.array(y)

        self._ridge_exceptions(X_t, y_t, lam, intercept)

        X_norm, y_norm = self._normalize(X_t,y_t)

        I = np.eye(X_t.shape[1])
        self._intercept = 0.0

        beta = np.linalg.solve(X_norm.T @ X_norm + lam*I, X_norm.T @ y_norm)

        X_std = np.std(X_t, axis = 0)
        X_safe_std = np.where(X_std == 0, 1.0, X_std)
        y_std = np.std(y_t)
        y_safe_std = np.where(y_std == 0, 1.0, y_std)

        beta = y_safe_std * beta/X_safe_std
        self._coef = beta

        if(intercept):
            self._intercept = np.mean(y_t) - np.mean(X_t, axis=0) @ beta
        return

    def _normalize(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Private class method for normalizing data.

        Scale the variables to have a mean of 0 and variance of 1.
        
        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The empirical target values.

        Returns:
            tuple[np.ndarray, np.ndarray]: The standardized arrays for X and y.
        """
        X_std = np.std(X, axis = 0)
        X_safe_std = np.where(X_std == 0, 1.0, X_std)
        y_std = np.std(y)
        y_safe_std = np.where(y_std == 0, 1.0, y_std)

        X_n = (X - np.mean(X, axis=0))/X_safe_std
        y_n = (y - np.mean(y))/y_safe_std
        return X_n, y_n

    def _ridge_exceptions(self, X: np.ndarray, y: np.ndarray, lam: float, intercept: bool) -> None:
        """Private method to log internal exceptions"""
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"Invalid data type: Expected feature matrix (X) to have a numeric data type, got '{X.dtype}'.")
        
        if not np.issubdtype(y.dtype, np.number):
            raise TypeError(f"Invalid data type: Expected target vector (y) to have a numeric data type, got '{y.dtype}'.")

        if not isinstance(lam, (float,int)):
            raise TypeError(f"Invalid data type: Expected lambda hyperparameter (lam) to have a float data type, got '{type(lam).__name__}'.")

        if not isinstance(intercept, bool):
            raise TypeError(f"Invalid data type: Expected intercept flag (intercept) to have a bool data type, got '{type(intercept).__name__}'.")

        if(lam<0):
            raise ValueError("Invalid value for (lam): Lambda hyperparameter (lam) must be a non-negative value.")
        
        if(X.ndim!=2):
            raise ValueError("Expected a 2-D array of shape (n_points, n_features).")
        
        if(y.ndim!=1):
            raise ValueError("Expected a 1-D array of shape (n_points, ).")
        
        if(X.shape[0]!=y.shape[0]):
            raise ValueError("Dimension mismatch: The number of rows in feature matrix (X) must equal the number of samples in the target vector (y).")
        return

class lasso(_linear_regression):
    """A public class for lasso linear regression.

    This class inherits the private class _linear_regression and add the fit method for lasso regression.
    """
    def fit(self, X: np.ndarray, y: np.ndarray, lam: float, intercept: bool = True, eps: float = 1e-4, max_iter: int = 1e4) -> None:
        """Fit the lasso model.
        
        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The empirical target values.
            lam (float): Regularization lasso hyperparameter. Should be a positive value. Higher values force the slope coefficients closer to 0.
            intercept (bool, optional): Decide whether to fit the model with an intercept coefficient or not.
            eps: (float, optional): Specify a convergence tolerance for the coordinate descent method.
            max_iter: (int, optional): The maximum number of iterations in the coordinate descent method to avoid infinity loops.

        Returns:
            None

        Attributes Updated:
            coef (np.ndarray): Calculated coefficients for linear regression.
            intercept (float, optional): Calculated intercept coefficient for linear regression.
        """
        X_t = np.array(X)
        y_t = np.array(y)

        self._lasso_exceptions(X_t, y_t, lam, intercept, eps, max_iter)

        X_norm, y_norm = self._normalize(X_t,y_t)
        self._intercept = 0.0

        beta = self._coordinate_descent(X_norm, y_norm, lam, eps, max_iter)

        X_std = np.std(X_t, axis = 0)
        X_safe_std = np.where(X_std == 0, 1.0, X_std)
        y_std = np.std(y_t)
        y_safe_std = np.where(y_std == 0, 1.0, y_std)

        beta = y_safe_std * beta/X_safe_std
        self._coef = beta

        if(intercept):
            self._intercept = np.mean(y_t) - np.mean(X_t, axis=0) @ beta
            self._intercept = self._intercept
        return

    def _coordinate_descent(self, X: np.ndarray, y: np.ndarray, lam: float, eps: float, max_iter: int) -> np.ndarray:
        """Private class method to calculate the coefficients for lasso regression.

        Implement an iterative method named coordinate descent to evaluate the lasso slope coefficients.
        See Delta project documentation for a more detailed explanation of the method.
        
        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The empirical target values.
            lam (float): Lasso hyperparameter.
            eps (float): A convergence threshold to decide when to stop the iterative loop.
            max_iter (int): Maximum number of iterations allowed to prevent infinite loops.

        Returns:
            np.ndarray: The calculated lasso slope coefficients.
        """
        p = X.shape[1]
        beta = np.zeros(p)
        beta_old = np.ones(p)
        iter = 0

        while(np.sum(np.abs(beta-beta_old))>eps or iter<max_iter):
            beta_old = beta.copy()
            for j in range(p):
                r_j = y - X @ beta + X[:,j]*beta[j]
                z_j = X[:,j]@r_j
                beta[j] = (np.sign(z_j)*np.maximum(0, np.abs(z_j)-lam/2))/X.shape[0]
                iter += 1

        return beta

    def _normalize(self, X: np.ndarray, y: np.ndarray) -> tuple[(np.ndarray, np.ndarray)]:
        """Private class method for normalizing data.

        Scale the variables to have a mean of 0 and variance of 1.
        
        Args:
            X (np.ndarray): The feature matrix.
            y (np.ndarray): The empirical target values.

        Returns:
            tuple[np.ndarray, np.ndarray]: The standardized arrays for X and y.
        """
        X_std = np.std(X, axis = 0)
        X_safe_std = np.where(X_std == 0, 1.0, X_std)
        y_std = np.std(y)
        y_safe_std = np.where(y_std == 0, 1.0, y_std)
        
        X_n = (X - np.mean(X, axis=0))/X_safe_std
        y_n = (y - np.mean(y))/y_safe_std

        return X_n, y_n

    def _lasso_exceptions(self, X: np.ndarray, y: np.ndarray, lam: float, intercept: bool, eps: float, max_iter: int) -> None:
        """Private method to log internal exceptions"""
        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"Invalid data type: Expected feature matrix (X) to have a numeric data type, got '{X.dtype}'.")
        
        if not np.issubdtype(y.dtype, np.number):
            raise TypeError(f"Invalid data type: Expected target vector (y) to have a numeric data type, got '{y.dtype}'.")

        if not isinstance(lam, (float,int)):
            raise TypeError(f"Invalid data type: Expected lambda hyperparameter (lam) to have a float data type, got '{type(lam).__name__}'.")

        if not isinstance(intercept, bool):
            raise TypeError(f"Invalid data type: Expected intercept flag (intercept) to have a bool data type, got '{type(intercept).__name__}'.")

        if not isinstance(eps, (float, int)):
            raise TypeError(f"Invalid data type: Expected epsilon convergence parameter (eps) to have a float data type, got '{type(eps).__name__}'.")

        if (not isinstance(max_iter, int)) and (not(isinstance(max_iter, float) and max_iter.is_integer())):
            raise TypeError(f"Invalid data type: Expected maximum of iterations convergence parameter (max_iter) to have a int data type, got '{type(max_iter).__name__}'.")

        if(lam<0):
            raise ValueError("Invalid value for (lam): Lambda hyperparameter (lam) must be a non-negative value.")

        if(eps<0):
            raise ValueError("Invalid value for (eps): Epsilon convergence parameter (eps) must be a non-negative value.")

        if(max_iter<=0):
            raise ValueError("Invalid value for (max_iter): Maximum of iterations convergence parameter (max_iter) must be a positive value.")
        
        if(X.ndim!=2):
            raise ValueError("Expected a 2-D array of shape (n_points, n_features).")
        
        if(y.ndim!=1):
            raise ValueError("Expected a 1-D array of shape (n_points, ).")
        
        if(X.shape[0]!=y.shape[0]):
            raise ValueError("Dimension mismatch: The number of rows in feature matrix (X) must equal the number of samples in the target vector (y).")
        return