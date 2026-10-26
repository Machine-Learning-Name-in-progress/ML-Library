import warnings

import numpy as np


class logistic_regression:
    """
    Logistic regression classification model

    Args:
        learn_rate (float): step size of gradient, that will be used in training method.
                            0.1 by default. Must be greater than 0.
        max_iter (int): max number of iterations on training process, preventing too many
                        iterations when the betas don't converge.
                        1000 by default. Must be greater than 0
        tol (float): convergence tolerance in model training. Iterations stop when the step
                     becomes lower than tolerance. 1e-6 by default. Must be greater than 0
    """

    def __init__(
        self, learn_rate: float = 0.1, max_iter: int = 1000, tol: float = 1e-6
    ):
        # learn_rate validate
        if not isinstance(learn_rate, (int, float)):
            raise TypeError("'learn_rate' must be a number")
        if learn_rate <= 0:
            raise ValueError("'learn_rate' must be greater than 0")

        # max_iter validate
        if not isinstance(max_iter, int):
            raise TypeError("'max_iter' must be an integer")
        if max_iter <= 0:
            raise ValueError("'max_iter' must be greater than 0")

        # tol validate
        if not isinstance(tol, (int, float)):
            raise TypeError("'tol' must be a number")
        if tol <= 0:
            raise ValueError("'tol' must be greater than 0")

        self._beta = None  # Can be set using train()
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

        for given values.

        Args:
            z (np.ndarray): compute sig(z) for each z in array

        Returns:
            (np.ndarray): array with sig(z_i) for all i in given z
        """

        return 1 / (1 + np.exp(-z))

    def predict_prob(self, x):
        """
        Predict the probability of x in class 1 using the current beta coefficients

        Args:
            x (np.ndarray): Feature matrix with shape (n_samples, n_features)

        Returns:
            (np.ndarray): Estimated probability of class 1 for each sample
        """
        # Validate training before predict
        if self._beta is None:
            raise ValueError("Cannot predict prob without training")

        x = np.asarray(x)
        x = self._add_ones_column(x)

        # Compute probability with given x and current coefficients beta
        z = x @ self._beta

        return self._sigmoid(z)

    def train(self, x_train, y_train):
        """
        Estimate the coefficients by maximizing the log-likelihood

        Args:
            x_train (np.ndarray): Matrix of training with shape (n_samples, n_features)
            y_train (np.ndarray): Correct results (0 or 1) for each sample of x_train
        """

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        # If we have classes other than (0, 1)
        if False in ((y_train == 0) | (y_train == 1)):
            raise ValueError("Logistic regression only works with two classes 0/1")

        # Validate dimensions of x and y (same number of rows)
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "Each point from x needs a result in y: same number of rows"
            )

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
        Predict the class (0 or 1) for new points. A point is classified as 1 when its
        estimated probability of class 1, given by predict_prob() is greater than 0.5, and
        classified as 0 otherwise.

        Args:
            x_new (np.ndarray): Feature matrix with shape (n_new_samples, n_features)

        Returns:
            (np.ndarray): array with predicted labels for each sample of x_new
        """

        probs = self.predict_prob(x_new)

        # According the plug-in method: select the most probable class
        # We have two classes, so p_0(x) = 1 - p_1(x) and p_1(x) > p_0(x) iff p_1(x) > 0.5
        prediction = probs > 0.5

        prediction = prediction.astype(int)

        return prediction
