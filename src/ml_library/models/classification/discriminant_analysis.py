import numpy as np
class _BaseDiscriminant:
    def __init__(self, reg_param: float = 0.0) -> None:
        if reg_param < 0 or reg_param > 1:
            raise ValueError("reg_param must be a value in [0,1]")

        self._priors = None
        self._means = None

        self.reg_param = reg_param # used for regularization by mixing sample_cov with the identity matrix:
                                    # (1 - r)*sample_cov + r * I
    def _regularize(self, cov):
        # we need only the variance not the covariance
        target = np.diag(np.diag(cov)) # extract the diagonal of the cov matrix 
                                        # as a 1d vector and then turns it into a diagonal matrix
        return (1 - self.reg_param) * cov + self.reg_param * target

    # calculates the common parts between the quadratic and linear analysis
    def _fit_common(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional (n_samples, n_features). got shape {X.shape}. Use X.reshape(1,-1) for a single point or X.reshape(-1, 1) for 1-D data.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional. got shape {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]} labels.")
            
        self.classes_, y_idx = np.unique(y, return_inverse=True) # classes gets only unique classes in increasing/alphabetic order,
                                                                    # y_idx allows to reconstruct y
        k = len(self.classes_)
        self._priors = np.bincount(y_idx, minlength=k)/len(y)
        self._means = np.array([X[y_idx == k].mean(axis=0) for k in range(k)])
        print(self._means)
        return y_idx

    def decision(self, X: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray):
        return self.classes_[np.argmax(self.decision(X), axis=1)]

    def predict_probabilities(self, X: np.ndarray):
        scores = self.decision(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

class QDA(_BaseDiscriminant):
    def __init__(self, reg_param: float = 0) -> None:
        super().__init__(reg_param)
        self._covariances = None

    @property
    def priors(self) -> np.ndarray:
        if self._priors is None:
            raise RuntimeError("Model is not yet fitted, fit first.")
        return self._priors

    @property
    def means(self) -> np.ndarray:
        if self._means is None:
            raise RuntimeError("Model is not yet fitted, fit first.")
        return self._means

    @property
    def covariances(self) -> list:
        if self._covariances is None:
            raise RuntimeError("Model is not yet fitted, fit first.")
        return self._covariances

    def fit(self, X: np.ndarray, y: np.ndarray):
        y_idx = self._fit_common(X,y)
        self._covariances = []

        self._L_inv = []
        self._log_det = []

        for k in range(len(self.classes_)):
            Xk = X[y_idx == k] - self.means[k]
            cov = Xk.T @ Xk / (len(Xk)-1)
            L = np.linalg.cholesky(cov)

            self._L_inv.append(np.linalg.inv(L)) #triangular and well-conditioned
            self._log_det.append(2.0 * np.sum(np.log(np.diag(L))))
            self._covariances.append(self._regularize(cov))
        return self

    @staticmethod
    def _mahalanobis_sq(X, mean, L_inv):
        Z = (X - mean) @ L_inv.T
        return np.sum(Z * Z, axis=1)

    def decision(self, X):
        n, K = X.shape[0], len(self.classes_)
        scores = np.empty((n, K))
        for k in range(K):
            scores[:, k] = (-0.5 * self._log_det[k]
                            - 0.5 * self._mahalanobis_sq(X, self.means[k], self._L_inv[k])
                            + np.log(self.priors[k]))
        return scores

class LDA(_BaseDiscriminant):
    def __init__(self, reg_param: float = 0) -> None:
        super().__init__(reg_param)
        self._coef = None
        self._intercept = None
        self._covariances = None

    @property
    def priors(self) -> np.ndarray:
        if self._priors is None:
            raise RuntimeError("Model is not yet fitted, fit first.")
        return self._priors

    @property
    def means(self) -> np.ndarray:
        if self._means is None:
            raise RuntimeError("Model is not yet fitted, fit first.")
        return self._means

    @property
    def coef(self) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("Model is not yet fitted, fit first.")
        return self._coef

    @property
    def intercept(self) -> np.ndarray:
        if self._intercept is None:
            raise RuntimeError("Model is not yet fitted, fit first.")
        return self._intercept

    @property
    def covariances(self) -> list:
        if self._covariances is None:
            raise RuntimeError("Model is not yet fitted, fit first.")
        return self._covariances
    
    def fit(self, X, y):
        y_idx = self._fit_common(X, y)
        centered = X - self.means[y_idx]
        cov = centered.T @ centered / (len(X) - len(self.classes_))
        cov = self._regularize(cov)

        self._covariances = cov
        self._coef = np.linalg.solve(cov, self.means.T).T
        self._intercept = (-0.5 * np.sum(self._coef * self._means, axis=1) + np.log(self.priors))

        return self

    def decision(self, X):
        return X @ self.coef.T + self._intercept

