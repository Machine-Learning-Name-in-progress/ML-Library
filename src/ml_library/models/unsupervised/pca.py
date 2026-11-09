import numpy as np

class PCA:
    """
    Class for the Principal Component Analysis (PCA) unsupervised model. 

    Args:
        n_components (int): Number of principal components.
        solver (str, optional): Method used to find the principal components.
            Options are 'eigen' (covariance matrix, default) and 'svd'.
        standardize (bool, optional): Defines whether the features must be standardized before fitting or not. 
            Default is False.
        
    Attributes:
        n_components (int): Number of principal components.
        components (np.ndarray): Principal components identified in the original feature space.
        mean (np.ndarray): Array containing the mean for each feature, estimated from the training set.
        std (np.ndarray): Array containing the standard deviation for each feature. 
            If standarize = False, std is None.
        evr (np.ndarray): Array containing the proportion of variance explained by each component.
    """
    def __init__(self, n_components: int, solver: str = 'eigen', standardize: bool = False) -> None:
        if not isinstance(n_components, (int, np.integer)):
            raise TypeError(f"The number of components must be an integer, got {type(n_components).__name__}.")

        if not isinstance(standardize, bool):
            raise TypeError(f"Standardize must be a boolean variable, got {type(standardize).__name__}.")

        if n_components <= 0:
            raise ValueError(f"The number of components must be positive, got {n_components}.")

        solvers = ['eigen', 'svd']
        if solver not in solvers:
            raise ValueError(f"Invalid solver: {solver}. Valid options are {solvers}.")
        
        self.n_components = n_components
        self.solver = solver
        self.standardize = standardize
        self.components = None
        self.mean = None
        self.std = None
        self.evr = None

    def _verify_input(self, X: np.ndarray) -> np.ndarray:
        """
        Private method to verify matrix inputs in fit() and transform().

        Args:
            X (np.ndarray): A 2-D array, where each row represents and entry, and each column represents a feature.

        Returns:
            np.ndarray
        """
        if not isinstance(X, (list, np.ndarray, tuple)):
            raise TypeError(f"Input must be array-like (list, numpy array or tuple). Got {type(X).__name__}.")

        try:
            X_array = np.asarray(X, dtype=np.float64)
        except (ValueError, TypeError):
            raise TypeError("Input must contain only numerical entries and be convertible to a numpy array.")

        if X_array.ndim != 2:
            raise ValueError(f"Expected a 2-D array of shape (n_samples, n_features), got {X_array.ndim}-D.")

        return X_array

    def fit(self, X: np.ndarray) -> None:
        """
        Runs the PCA algorithm on a matrix, obtaining the principal components.

        Args:
            X (np.ndarray): A 2-D array of training data, where each row represents an entry, and each column represents a feature.
        """
        X_array = self._verify_input(X)
        
        _, n_feat = X_array.shape
        if self.n_components > n_feat:
            raise ValueError(f"The number of components ({self.n_components}) cannot be greater than the number of features {n_feat}.")
        
        self.mean = np.mean(X_array, axis=0)
        X_center = X_array - self.mean

        if self.standardize:
            self.std = np.std(X_center, axis=0)
            X_center[:, self.std != 0] = X_center[:, self.std != 0] / self.std[self.std != 0]

        if self.solver == 'eigen':
            # Calculates the eigenvectors of the covariance matrix.
            cov_matrix = np.cov(X_center, rowvar=False)
            e_val, e_vect = np.linalg.eigh(cov_matrix)
            # The order of precedence for the eigenvectors is defined by the associated eigenvalue.
            ind_vect = np.argsort(e_val)[::-1]
            e_vect = e_vect[:, ind_vect]
            e_val_sort = e_val[ind_vect]
            self.components = e_vect[:, :self.n_components]
            self.evr = e_val_sort[:self.n_components] / np.sum(e_val_sort)
        else:
            # Calculates the SVD decomposition (as U \Sigma V), and obtains the first components in V.
            _, E, V_transp = np.linalg.svd(X_center, full_matrices=False)
            V = V_transp.T
            self.components = V[:, :self.n_components]
            self.evr = (E[:self.n_components] ** 2) / np.sum(E ** 2)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the projection of the data from the original features onto the principal components.

        Args:
            X (np.ndarray): A 2-D array, where each row represents an entry, and each column represents a feature.

        Returns:
            np.ndarray
        """
        if self.components is None:
            raise RuntimeError("Model is not fitted yet, call fit() first.")

        X_array = self._verify_input(X)
        X_center = X_array - self.mean
        
        if self.standardize:
            X_center[:, self.std != 0] = X_center[:, self.std != 0] / self.std[self.std != 0]
        
        return np.dot(X_center, self.components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the fit(), then the transform() method to project the data onto the principal components.

        Args:
            X (np.ndarray): A 2-D array of training data, where each row represents an entry, and each column represents a feature.
        
        Returns:
            np.ndarray
        """
        self.fit(X)
        return self.transform(X)