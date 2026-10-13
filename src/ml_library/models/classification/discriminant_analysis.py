import numpy as np
from ml_library.utils import _solvers

class _BaseDiscriminant:
    def __init__(self, solver: str = "cholesky", reg_param: float = 0.0) -> None:
        if solver not in _solvers.solvers:
            raise ValueError(f"Selected solver does not exist. Availables solvers are: {_solvers.solvers}")

        if reg_param < 0 or reg_param > 1:
            raise ValueError("reg_param must be a value in [0,1]")
        
        self.solver = solver
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
            raise ValueError(f"X must be 2-dimensional (n_samples, n_features). got shape {X.shape}.")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional. got shape {y.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]} labels.")
            
        self.classes_, y_idx = np.unique(y, return_inverse=True) # classes gets only unique classes in increasing/alphabetic order,
                                                                    # y_idx allows to reconstruct y
        k = len(self.classes_)
        self.priors_ = np.bincount(y_idx, minlength=k)/len(y)
        self.means_ = np.array([X[y_idx == k].mean(axis=0) for k in range(k)])
        print(self.means_)
        return y_idx

    def decision(self, X: np.ndarray):
        raise NotImplementedError

    def predict(self, X: np.ndarray):
        return self.classes_[np.argmax(self.decision(X), axis=1)]

    def predict_probabilities(self, X: np.ndarray):
        pass

class QDA(_BaseDiscriminant):
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    def decision(self, X):
        return super().decision(X)

class LDA(_BaseDiscriminant):
    def fit(self, X, y):
        pass

    def decision(self, X):
        return super().decision(X)

if __name__ == "__main__":
    from ml_library.utils._random import get_rng

    B = _BaseDiscriminant()
    rng = get_rng(0)

    X = rng.normal(size=(40,2))
    y = rng.integers(0,2,size=40)

    B._fit_common(X,y)

