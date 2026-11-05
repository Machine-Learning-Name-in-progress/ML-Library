import numpy as np
class _BaseDiscriminant:
    def __init__(self, reg_param: float = 0.0) -> None:
        if reg_param < 0 or reg_param > 1:
            raise ValueError("reg_param must be a value in [0,1]")

        self._priors = None
        self._means = None
        self._classes = None
        self._elements_per_class = None

        self.reg_param = reg_param # used for regularization by mixing sample_cov with the diagonal matrix:
                                    # (1 - r)*sample_cov + r * I
    
    @property
    def elements_per_class(self) -> np.ndarray:
        if self._elements_per_class is None:
            raise RuntimeError("Model is not yet fitted, fit first.")
        return self._elements_per_class

    @property
    def classes(self) -> np.ndarray:
        if self._classes is None:
            raise RuntimeError("Model is not yet fitted, fit first.")
        return self._classes

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
            
        self._classes, y_idx = np.unique(y, return_inverse=True) # classes gets only unique classes in increasing/alphabetic order,
                                                                    # y_idx allows to reconstruct y
        k = len(self.classes)
        self._elements_per_class = np.bincount(y_idx, minlength=k)
        self._priors = self.elements_per_class/len(y)
        self._means = np.array([X[y_idx == i].mean(axis=0) for i in range(k)])
        return y_idx

    def decision(self, X: np.ndarray):
        """
        Returns the scores for each point, a higher score means 
        it's more likely to belong to that class. Use `predict(X)`
        to get the classes that maximizes these scores. The only
        meaning this has is difference between classes, scores are 
        not compatible between models.

        Args:
            X (array-like): Feature matrix with shape (n_samples, n_features).

        Returns:
            (np.ndarray): Matrix with shape (n_samples, n_classes). Column k holds
                        the score of class self.classes[k].

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray | list):
        """
        Predict the most probable class for each sample.

        Args:
            X (array-like): Feature matrix with shape (n_samples, n_features).

        Returns:
            (np.ndarray): Array with shape (n_samples,) holding the predicted label of
                        each sample, taken from self.classes.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        X = np.asarray(X)

        return self.classes[np.argmax(self.decision(X), axis=1)]

    def predict_probabilities(self, X: np.ndarray | list):
        """
        Estimate the posterior probability of each class for each sample.

        Computed as a softmax over the scores from decision(). The class priors are
        the class frequencies in the training data.

        Args:
            X (array-like): Feature matrix with shape (n_samples, n_features).

        Returns:
            (np.ndarray): Matrix with shape (n_samples, n_classes). Entry [i, k] is
                        P(class self.classes[k] | X[i]); each row sums to 1.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """

        X = np.asarray(X)

        # Maybe we can put the softmax function on a library wide
        #  module along sigmoid and other similar functions
        scores = self.decision(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

class QDA(_BaseDiscriminant):
    """
    Quadratic discriminant analysis class

    Methods:
        fit
        predict
        predict_probabilities
        decision
    
    Properties:
        covariances
        means
        priors
        elements_per_class
        classes
    """
    def __init__(self, reg_param: float = 0) -> None:
        super().__init__(reg_param)
        self._covariances = None

    @property
    def covariances(self) -> list:
        if self._covariances is None:
            raise RuntimeError("Model is not yet fitted, fit first.")
        return self._covariances

    def fit(self, X: np.ndarray | list, y: np.ndarray | list):
        """
        Fit the model: estimate the prior, mean and covariance matrix of each class.

        Each class covariance is the sample covariance (divided by n_k - 1), shrunk
        toward its diagonal by reg_param.

        Args:
            X (array-like): Training data with shape (n_samples, n_features).
            y (array-like): Class label of each sample, with shape (n_samples,).
                            Labels can be any sortable values (numbers or strings).

        Returns:
            QDA: The fitted model (self).

        Raises:
            ValueError: If X is not 2-D, y is not 1-D, their lengths differ, or a
                        class has fewer than 2 samples.
            numpy.linalg.LinAlgError: If a class covariance is singular, e.g. a class
                        has no more samples than features with reg_param = 0, or a
                        feature is constant within a class.

        If an error is raised, the model must be fitted again before use because i was
        lazy to treat the following edge case: you fit the model once without errors but
        then you fit it again with a dataset with faulty parameters. Some of the self
        variables are assigned again before it throws an error so you end up with self 
        variables from the first dataset and the second dataset mixed in the same model,
        and the predictions won't make sense.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        y_idx = self._fit_common(X,y)

        if (self.elements_per_class < 2).any():
            raise ValueError(f"All classes must have at least 2 samples. Class(es) {self.classes[self.elements_per_class < 2]} have less than 2.")

        self._covariances = []

        self._L_inv = []
        self._log_det = []

        for k in range(len(self.classes)):
            Xk = X[y_idx == k] - self.means[k]
            cov = self._regularize(Xk.T @ Xk / (len(Xk)-1))
            L = np.linalg.cholesky(cov)

            self._L_inv.append(np.linalg.inv(L)) #triangular and well-conditioned
            self._log_det.append(2.0 * np.sum(np.log(np.diag(L))))
            self._covariances.append(cov)
        return self

    @staticmethod
    def _mahalanobis_sq(X, mean, L_inv):
        Z = (X - mean) @ L_inv.T
        return np.sum(Z * Z, axis=1)

    # i don't remember exactly the deduction i did to get
    #  to these formulas because it was days ago, but i remember
    #  i started from scikit-learn's documentation which had some
    #  of the math
    def decision(self, X: np.ndarray | list):
        X = np.asarray(X)

        n, K = X.shape[0], len(self.classes)
        scores = np.empty((n, K))
        for k in range(K):
            scores[:, k] = (-0.5 * self._log_det[k]
                            - 0.5 * self._mahalanobis_sq(X, self.means[k], self._L_inv[k])
                            + np.log(self.priors[k]))
        return scores

class LDA(_BaseDiscriminant):
    """
    Quadratic discriminant analysis class

    Methods:
        fit
        predict
        predict_probabilities
        decision
    
    Properties:
        coef
        intercept
        covariances
        means
        priors
        elements_per_class
        classes
    """
    def __init__(self, reg_param: float = 0) -> None:
        super().__init__(reg_param)
        self._coef = None
        self._intercept = None
        self._covariances = None

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
    def covariances(self) -> np.ndarray:
        if self._covariances is None:
            raise RuntimeError("Model is not yet fitted, fit first.")
        return self._covariances
    
    def fit(self, X: np.ndarray | list, y: np.ndarray | list):
        """
        Fit the model: estimate the prior and mean of each class, and one covariance
        matrix shared by all classes.

        The shared covariance is the pooled within-class covariance (divided by
        n_samples - n_classes), shrunk toward its diagonal by reg_param.

        Args:
            X (array-like): Training data with shape (n_samples, n_features).
            y (array-like): Class label of each sample, with shape (n_samples,).
                            Labels can be any sortable values (numbers or strings).

        Returns:
            LDA: The fitted model (self).

        Raises:
            ValueError: If X is not 2-D, y is not 1-D, their lengths differ, or there
                        are not more samples than classes.
            numpy.linalg.LinAlgError: If the pooled covariance is singular, e.g. a
                        feature is constant within every class.

        If an error is raised, the model must be fitted again before use because i was
        lazy to treat the following edge case: you fit the model once without errors but
        then you fit it again with a dataset with faulty parameters. Some of the self
        variables are assigned again before it throws an error so you end up with self 
        variables from the first dataset and the second dataset mixed in the same model,
        and the predictions won't make sense.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        y_idx = self._fit_common(X, y)

        if X.shape[0] <= self.classes.shape[0]:
            raise ValueError("Number of samples must be greater than number of classes.")
        

        centered = X - self.means[y_idx]
        cov = centered.T @ centered / (len(X) - len(self.classes))
        cov = self._regularize(cov)

        self._covariances = cov
        self._coef = np.linalg.solve(cov, self.means.T).T
        self._intercept = (-0.5 * np.sum(self.coef * self.means, axis=1) + np.log(self.priors))

        return self

    def decision(self, X: np.ndarray | list):
        X = np.asarray(X)

        return X @ self.coef.T + self._intercept

