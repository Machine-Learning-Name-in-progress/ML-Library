import numpy as np

class _linear_regression:
    def __init__(self):
        self.coef = []
        self.intercept = 0
        return
    
    def train(self):
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_p = np.array(X)
        y = X_p @ self.coef + self.intercept
        return y

    def RSS(self, X: np.ndarray, y: np.ndarray) -> float:
        y_r = np.array(y)
        y_hat = self.predict(X)
        rss = np.sum((y_r-y_hat)**2)
        return rss

    def MSE(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.RSS(X,y)/len(y)

    def RMSE(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.sqrt(self.MSE(X,y))

    def R2(self, X: np.ndarray, y: np.ndarray) -> float:
        y_r = np.array(y)
        y_hat = self.predict(X)

        r2 = 1 - (np.sum((y_r-y_hat)**2))/(np.sum((y_r-np.mean(y_r))**2))
        return r2

class ordinary_least_squares(_linear_regression):
    def train(self, X: np.ndarray, y: np.ndarray, intercept: bool = True) -> None:
        X_t = np.array(X)
        y_t = np.array(y)
        self.intercept = 0

        if(intercept):
            ones_column = np.ones((X_t.shape[0],1))
            X_t = np.hstack((ones_column, X_t))

        beta = np.linalg.solve(X_t.T @ X_t, X_t.T @ y_t)

        if(intercept):
            self.intercept = beta[0]
            beta = beta[1:]

        self.coef = beta
        return

class ridge(_linear_regression):
    def train(self, X: np.ndarray, y: np.ndarray, lam: float, intercept: bool = True) -> None:
        X_t = np.array(X)
        y_t = np.array(y)
        X_norm, y_norm = self._normalize(X_t,y_t)

        I = np.eye(X_t.shape[1])
        self.intercept = 0

        beta = np.linalg.solve(X_norm.T @ X_norm + lam*I, X_norm.T @ y_norm)
        beta = np.std(y_t) * beta/np.std(X_t, axis = 0)
        self.coef = beta

        if(intercept):
            self.intercept = np.mean(y_t) - np.mean(X_t, axis=0) @ beta
        return

    def _normalize(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_n = (X - np.mean(X, axis=0))/np.std(X, axis = 0)
        y_n = (y - np.mean(y))/np.std(y)
        return X_n, y_n

class lasso(_linear_regression):
    def train(self, X: np.ndarray, y: np.ndarray, lam: float, intercept: bool = True, eps: float = 1e-4, max_iter: int = 1e4) -> None:
        X_t = np.array(X)
        y_t = np.array(y)
        X_norm, y_norm = self._normalize(X_t,y_t)
        self.intercept = 0

        beta = self._cordinate_descent(X_norm, y_norm, lam, eps, max_iter)

        beta = np.std(y_t) * beta/np.std(X_t, axis = 0).reshape(X_t.shape[1],1)
        self.coef = beta.flatten()

        if(intercept):
            self.intercept = np.mean(y_t) - np.mean(X_t, axis=0) @ beta
        return

    def _cordinate_descent(self, X: np.ndarray, y: np.ndarray, lam: float, eps: float, max_iter: int) -> np.ndarray:
      p = X.shape[1]
      beta = np.zeros(p).reshape(p,1)
      beta_old = np.ones(p).reshape(p,1)
      y = y[:, np.newaxis]
      iter = 0

      while(np.sum(np.abs(beta-beta_old))>eps or iter<max_iter):
          beta_old = beta
          R = y + X*beta.T - X @ beta
          z = np.sum(X*R, axis=0).reshape(p,1)
          beta = (np.sign(z)*np.maximum(0, np.abs(z)-lam/2))/X.shape[0]
          iter += 1

      return beta

    def _normalize(self, X: np.ndarray, y: np.ndarray) -> tuple[(np.ndarray, np.ndarray)]:
        X_n = (X - np.mean(X, axis=0))/np.std(X, axis = 0)
        y_n = (y - np.mean(y))/np.std(y)
        return X_n, y_n