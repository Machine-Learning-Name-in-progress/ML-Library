import warnings

import numpy as np


class LogisticRegression:
    """
    Logistic regression classification model for two classes, trained by gradient
    ascent on the log-likelihood.

    Need to standardize the features (x) before calling fit() to
    prevent exp(-z) overflow.

    The labels can be any two values (e.g. 0/1, -1/1, "no"/"yes"). They are sorted
    (numerically or alphabetically) and the second one is the positive class.

    Args:
        learn_rate (float): step size of gradient in each update. Larger values converge
                            faster, but a value too large makes the training oscillate.
                            0.5 by default. Must be greater than 0.
        max_iter (int): max number of iterations on training process. If reached before
                        convergence, a RuntimeWarning is raised and the last betas are
                        kept. 5000 by default. Must be greater than 0
        tol (float): convergence tolerance in model training. Iterations stop when the step
                     becomes lower than tolerance. 1e-6 by default. Must be greater than 0
    """

    def __init__(
        self, learn_rate: float = 0.5, max_iter: int = 5000, tol: float = 1e-6
    ):
        # learn_rate validate (bool is a subclass of int, so it is rejected explicitly)
        if isinstance(learn_rate, bool) or not isinstance(
            learn_rate, (int, float, np.integer, np.floating)
        ):
            raise TypeError("'learn_rate' must be a number")
        if learn_rate <= 0:
            raise ValueError("'learn_rate' must be greater than 0")

        # max_iter validate
        if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
            raise TypeError("'max_iter' must be an integer")
        if max_iter <= 0:
            raise ValueError("'max_iter' must be greater than 0")

        # tol validate
        if isinstance(tol, bool) or not isinstance(
            tol, (int, float, np.integer, np.floating)
        ):
            raise TypeError("'tol' must be a number")
        if tol <= 0:
            raise ValueError("'tol' must be greater than 0")

        self._beta = None  # Can be set using fit()
        self._classes = None  # Can be set using fit(): the two sorted labels
        self._learn_rate = learn_rate
        self._max_iter = max_iter
        self._tol = tol

    def _add_ones_column(self, x):
        """
        Add a column of 1s to the given x. Essential for compute beta_0 in training
        process

        Args:
            x (np.ndarray): Matrix with shape (n_samples, n_features)

        Returns:
            (np.ndarray): Matrix with shape (n_samples, n_features + 1)
        """

        # Create an array with (n_sample) ones
        b_0 = np.ones(x.shape[0])

        # Concatenate
        return np.column_stack((b_0, x))

    def _sigmoid(self, z):
        """
        Computes the value of sigmoid function:
        sig(z) = 1 / (1 + exp(-z))

        Args:
            z (np.ndarray): compute sig(z) for each z in array

        Returns:
            (np.ndarray): array with sig(z_i) for all i in given z
        """

        return 1 / (1 + np.exp(-z))

    def predict_probabilities(self, x):
        """
        Predict the probability of each class for x using the current beta
        coefficients

        Args:
            x (np.ndarray): Feature matrix with shape (n_samples, n_features)

        Returns:
            (np.ndarray): Matrix with shape (n_samples, 2). Column k has the estimated
                          probability of the k-th sorted label (column 1 is the
                          positive class)
        """
        # Validate training before predict
        if self._beta is None:
            raise ValueError("Cannot predict before training. Use '.fit()' first")

        x = np.asarray(x)
        x = self._add_ones_column(x)

        # Compute probability with given x and current coefficients beta
        z = x @ self._beta
        prob_1 = self._sigmoid(z)

        # Knowing that p_0(x) = 1 - p_1(x)
        return np.column_stack((1 - prob_1, prob_1))

    def fit(self, x_train, y_train):
        """
        Estimate the coefficients by maximizing the log-likelihood

        Args:
            x_train (np.ndarray): Matrix of training with shape (n_samples, n_features)
            y_train (np.ndarray): Labels with shape (n_samples). Must have exactly two
                                  distinct values
        """

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        # y must be a vector with shape (n_samples)
        if y_train.ndim != 1:
            raise ValueError("y must be a 1D array with shape (n_samples)")

        # Validate dimensions of x and y (same number of rows)
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "Each point from x needs a result in y: same number of rows"
            )

        # Sorted labels, and the position of each y in it: y_train = classes[y_index]
        classes, y_index = np.unique(y_train, return_inverse=True)

        # Logistic regression works with exactly two classes
        if len(classes) != 2:
            raise ValueError(
                f"Logistic regression needs exactly 2 classes ({len(classes)} given)"
            )

        self._classes = classes

        # Train with 0 for classes[0] and 1 for classes[1]
        y_train = y_index

        x_train = self._add_ones_column(x_train)
        N, p = x_train.shape

        # Start with beta = 0
        beta = np.zeros(p)

        # Repeat the loop up to max_iter times
        for _ in range(self._max_iter):
            # Compute probability for given x_train, and the error
            z = x_train @ beta
            prob = self._sigmoid(z)
            error = y_train - prob

            # Compute the gradient
            gradient = (np.transpose(x_train) @ error) / N

            # Set the step based on learn_rate and gradient direction
            step = self._learn_rate * gradient

            # Beta for next iteration
            beta = beta + step

            # If the step is very tiny, we can stop
            if np.linalg.norm(step) < self._tol:
                break

        # If we reach the max_iter iterations, then save the last coefficients and advise
        else:
            warnings.warn(
                f"Did not converge in max_iter = {self._max_iter} iterations",
                RuntimeWarning,
            )

        self._beta = beta

    def predict(self, x_new):
        """
        Predict the label for new points. A point receives the positive label
        (classes[1]) when its estimated probability, given by predict_probabilities(),
        is greater than 0.5, and the other label (classes[0]) otherwise.

        Args:
            x_new (np.ndarray): Feature matrix with shape (n_new_samples, n_features)

        Returns:
            (np.ndarray): array with predicted labels for each sample of x_new
        """

        # Probability of the positive class (column 1)
        probs = self.predict_probabilities(x_new)[:, 1]

        # According the plug-in method: select the most probable class
        # We have two classes, so p_0(x) = 1 - p_1(x) and p_1(x) > p_0(x) iff p_1(x) > 0.5
        prediction = probs > 0.5

        prediction = prediction.astype(int)

        # Index 0 or 1 -> original label
        return self._classes[prediction]
