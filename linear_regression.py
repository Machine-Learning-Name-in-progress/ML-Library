import numpy as np

class linear_regression:
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

class ordinary_least_squares(linear_regression):
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

class ridge(linear_regression):
    def train(self, X: np.ndarray, y: np.ndarray, lam: float, intercept: bool = True) -> None:
        X_t = np.array(X)
        y_t = np.array(y)
        I = np.eye(X_t.shape[1])
        self.intercept = 0

        if(intercept):
            ones_column = np.ones((X_t.shape[0],1))
            X_t = np.hstack((ones_column, X_t))

            zeros_column = np.zeros((I.shape[0],1))
            I = np.hstack((zeros_column, I))

            zeros_line = np.zeros((1,I.shape[1]))
            I = np.vstack((zeros_line, I))

        beta = np.linalg.solve(X_t.T @ X_t + lam*I, X_t.T @ y_t)

        if(intercept):
            self.intercept = beta[0]
            beta = beta[1:]
            
        self.coef = beta
        return