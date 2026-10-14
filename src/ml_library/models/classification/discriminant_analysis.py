import numpy as np
class _BaseDiscriminant:
    def __init__(self, reg_param: float = 0.0) -> None:
        if reg_param < 0 or reg_param > 1:
            raise ValueError("reg_param must be a value in [0,1]")

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
        scores = self.decision(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

class QDA(_BaseDiscriminant):
    def fit(self, X: np.ndarray, y: np.ndarray):
        y_idx = self._fit_common(X,y)
        self.covariances_ = []

        self._L_inv = []
        self._log_det = []

        for k in range(len(self.classes_)):
            Xk = X[y_idx == k] - self.means_[k]
            cov = Xk.T @ Xk / len(Xk - 1)
            L = np.linalg.cholesky(cov)

            self._L_inv.append(np.linalg.inv(L)) #triangular and well-conditioned
            self._log_det.append(2.0 * np.sum(np.log(np.diag(L))))
            self.covariances_.append(self._regularize(cov))
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
                            - 0.5 * self._mahalanobis_sq(X, self.means_[k], self._L_inv[k])
                            + np.log(self.priors_[k]))
        return scores

class LDA(_BaseDiscriminant):
    def fit(self, X, y):
        y_idx = self._fit_common(X, y)
        centered = X - self.means_[y_idx]
        cov = centered.T @ centered / (len(X) - len(self.classes_))
        cov = self._regularize(cov)
        self.coef_ = np.linalg.solve(cov, self.means_.T).T
        self.intercept_ = (-0.5 * np.sum(self.coef_ * self.means_, axis=1) + np.log(self.priors_))

        return self

    def decision(self, X):
        return X @ self.coef_.T + self.intercept_

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ml_library.utils._random import get_rng

    rng = get_rng(0)

    # Three Gaussian classes with different means and covariances
    means = [np.array([0.0, 0.0]), np.array([3.0, 3.0]), np.array([0.0, 4.0])]
    covs = [
        np.array([[1.0, 0.8], [0.8, 1.0]]),    # stretched along the diagonal
        np.array([[0.3, 0.0], [0.0, 2.0]]),    # tall and narrow
        np.array([[2.0, -0.5], [-0.5, 0.5]]),  # wide
    ]
    n_per_class = 100

    X = np.vstack([rng.multivariate_normal(m, c, size=n_per_class) for m, c in zip(means, covs)])
    y = np.repeat(np.arange(len(means)), n_per_class)

    # Grid covering the data, to color each point of the plane by its predicted class
    pad = 1.0
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    K = len(means)

    for ax, model, name in zip(axes, [LDA(), QDA()], ["LDA", "QDA"]):
        model.fit(X, y)
        Z = model.predict(grid).reshape(xx.shape)
        acc = np.mean(model.predict(X) == y)

        ax.contourf(xx, yy, Z, levels=np.arange(K + 1) - 0.5, cmap="Pastel1", alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="Set1", edgecolor="k", s=20)
        ax.scatter(model.means_[:, 0], model.means_[:, 1], c="black", marker="X", s=150, label="class means")
        ax.set_title(f"{name}  (train accuracy: {acc:.2%})")
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

