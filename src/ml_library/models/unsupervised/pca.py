import numpy as np

# TODO: review and add docstrings, exceptions 
# TODO: optional standardization parameter, consider SVD method implementation
# TODO: remove temporary tests
class pca:
    """
        Class for the Principal Component Analysis (PCA) unsupervised model. 

        Args:
            n_components (int): Number of principal components.
        
        Attributes:
            n_components (int): Number of principal components.
            components (np.ndarray): Principal components identified in the original feature space.
            mean (np.ndarray): Array containing the mean for each feature, estimated from the training set.
    """
    def __init__(self, n_components: int) -> None:
        if not isinstance(n_components, int):
            raise TypeError(f"The number of components must be an integer, got {type(n_components).__name__}.")

        if n_components <= 0:
            raise ValueError(f"The number of components must be positive, got {n_components}.")
        
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X: np.ndarray) -> None:
        """
        Runs the PCA algorithm on a matrix, obtaining the principal components.

        Uses eigenvectors for the covariance matrix to identify the directions of the components.

        Args:
            X (np.ndarray): A 2-D array of training data, where each row represents an entry, and each column represents a feature.
        """
        if X.ndim != 2:
            raise ValueError(f"Expected a 2-D array of shape (n_samples, n_features), got {X.ndim}-D.")
        _, n_feat = X.shape
        if self.n_components > n_feat:
            raise ValueError(f"The number of components ({self.n_components}) cannot be greater than the number of features {n_feat}.")
        
        self.mean = np.mean(X, axis=0)
        X_center = X - self.mean

        cov_matrix = np.cov(X_center, rowvar=False)

        e_val, e_vect = np.linalg.eigh(cov_matrix)

        # The order of precedence for the eigenvectors is defined by the associated eigenvalue.
        ind_vect = np.argsort(e_val)[::-1]
        e_vect = e_vect[:, ind_vect]
        self.components = e_vect[:, :self.n_components]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the projection of the data from the original features onto the principal components.

        Args:
            X (np.ndarray): A 2-D array, where each row represents an entry, and each column represents a feature.
        """
        if self.components is None:
            raise RuntimeError("Model is not fitted yet, call fit() first.")
        X_center = X - self.mean
        return np.dot(X_center, self.components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the fit(), then the transform() method to project the data onto the principal components.

        Args:
            X (np.ndarray): A 2-D array of training data, where each row represents an entry, and each column represents a feature.
        """
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
    PCA = pca(n_components="67")
    X_red = PCA.fit_transform(X)
    print(X)
    print(PCA.mean)
    print(PCA.components)
    print(X_red)
    print(X_red.shape)