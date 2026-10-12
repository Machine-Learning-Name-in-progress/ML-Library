import numpy as np

class PolynomialRegression:
    # TODO: add lasso and ridge
    def __init__(self, degree, intercept=True) -> None:
        self._degree = degree
        self._intercept = intercept

    def _build_design(self, x: np.ndarray) -> np.ndarray:
        X = np.array([[val_xi**d for d in range(1, self._degree + 1)] for val_xi in x])

        if self._intercept:
            X = np.column_stack((np.ones(shape=(len(x), 1)), X))

        return X
        
    def train(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = self._build_design(x)
        coef = np.linalg.solve(X.T @ X, X.T @ y)
        self.coef = coef
        return coef

    def predict(self, x: np.ndarray) -> np.ndarray:
        X = self._build_design(x)
        preds = X @ self.coef
        return preds

def RSS(y: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum((y - y_pred)**2)

def MSE(y: np.ndarray, y_pred: np.ndarray) -> float:
    return RSS(y, y_pred)/len(y)

def RMSE(y: np.ndarray, y_pred: np.ndarray) -> float:
    pass

def MAE(y: np.ndarray, y_pred: np.ndarray) -> float:
    pass

def R2(y: np.ndarray, y_pred: np.ndarray) -> float:
    pass

def R2_adj(y: np.ndarray, y_pred: np.ndarray) -> float:
    pass