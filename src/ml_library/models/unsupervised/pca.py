import numpy as np

# TODO: docstrings, exceptions, optional standardization parameter, consider SVD method
# TODO: remove temporary tests
class pca:
    """
        Class for Principal Component Analysis. 
        Uses eigenvectors for the covariance matrix (symmetric) to identify the directions.
        The order of precedence for the eigenvectors is defined by the associated eigenvalue.
        Receives a matrix where each row represents a single entry, and the columns represent the features.
    """
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_center = X - self.mean

        cov_matrix = np.cov(X_center, rowvar=False)

        e_val, e_vect = np.linalg.eigh(cov_matrix)
        ind_vect = np.argsort(e_val)[::-1]
        e_vect = e_vect[:, ind_vect]
        self.components = e_vect[:, :self.n_components]

    def transform(self, X):
        X_center = X - self.mean
        return np.dot(X_center, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

if __name__ == "__main__":
    np.random.seed(67)
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10]
    ])
    PCA = pca(n_components=1)
    X_red = PCA.fit_transform(X)
    print(X)
    print(PCA.mean)
    print(PCA.components)
    print(X_red)
    print(X_red.shape)