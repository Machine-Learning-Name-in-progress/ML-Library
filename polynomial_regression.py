import numpy as np
from linear_regression import ordinary_least_squares, ridge, lasso

class polynomial_regression:
    def __init__(self, degree: int, model_choice: str = 'ols', intercept=True, lam: float = 1.0, eps: float = 1e-4, max_iter: int = 1e4) -> None:
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
        else:
            raise ValueError("model_choice must be 'ols', 'ridge' or 'lasso'")

    def _build_design(self, x: np.ndarray) -> np.ndarray:
        X = np.array([[val_xi**d for d in range(1, self.degree + 1)] for val_xi in x])
        return X
        
    def train(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
    # TODO: remove temporary tests
    x_teste = np.array([-2, -1, 0, 1, 2])
    y_teste = 3 + 2*x_teste + x_teste**2
    np.random.seed(67)
    erro = np.random.normal(0.0, 0.5, size=len(x_teste))
    y_ruido = y_teste + erro
    grau = 4

    # Teste OLS
    teste_ols = polynomial_regression(degree=grau, model_choice='ols', intercept=True)
    teste_ols.train(x_teste, y_ruido)
    print(f"Coeficientes OLS: {teste_ols.model_intercept:.3f} e {teste_ols.coef}")

    # Teste Ridge
    teste_ridge = polynomial_regression(degree=grau, model_choice='ridge', intercept=True, lam=0.1)
    teste_ridge.train(x_teste, y_ruido)
    print(f"Coeficientes Ridge: {teste_ridge.model_intercept:.3f} e {teste_ridge.coef}")

    # Teste Lasso
    teste_lasso = polynomial_regression(degree=grau, model_choice='lasso', intercept=True, lam=0.1)
    teste_lasso.train(x_teste, y_ruido)
    print(f"Coeficientes Lasso: {teste_lasso.model_intercept:.3f} e {teste_lasso.coef}")