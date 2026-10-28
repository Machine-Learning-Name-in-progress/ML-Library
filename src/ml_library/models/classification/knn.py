import numpy as np


class KNN:
    """
    K-Nearest Neighbors (KNN) classification method

    Args:
        k (int): Number of neighbors considered for prediction
    """

    def __init__(self, k: int) -> None:

        # K validation: K negative or not int parameter
        if not isinstance(k, (int, np.integer)) and not (
            isinstance(k, float) and k.is_integer()
        ):
            raise TypeError("K must be an integer")
        elif k < 1:
            raise ValueError("K must be an integer greater than 0")

        self._k = int(k)  # Hyperparameter K
        self._x_train = None  # Can be set using fit()
        self._y_train = None  # Can be set using fit()

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using the training datasets

        Args:
            x (np.ndarray): Training features matrix with shape (n_samples, n_features)
            y (np.ndarray): Classifications vector with shape (n_samples)
        """

        # Convert lists, DataFrames and Series to NumPy arrays
        x = np.asarray(x)
        y = np.asarray(y)

        # Dimension validate
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x has {x.shape[0]} samples while y has {y.shape[0]}. Both need to be equal"
            )

        # Number of points validation
        if len(x) < self._k:
            raise ValueError(
                f"KNN model needs at least K = {self._k} points to work ({len(x)} passed)"
            )

        self._x_train = x  # Predictors
        self._y_train = y  # Categories

    def _predict_point(self, x_new: np.ndarray):
        """
        Predict a classification for a new data point

        Args:
            x_new (np.ndarray): New point to classify with shape (n_features)

        Returns:
            Predicted classification for the new point
        """

        # Compute k-nearest distances
        distances = [np.linalg.norm(x_new - x_train) for x_train in self._x_train]

        # Pick the k-nearest points (index of k-shortest distances)
        nearest = np.argsort(distances)[: self._k]

        # Captures the classification of the nearest points
        nearest_categories = self._y_train[nearest]

        # The classes and counts of the nearest points
        classes, counts = np.unique(nearest_categories, return_counts=True)

        # Choose the most frequent classification
        biggest_index = np.argmax(counts)
        classification = classes[biggest_index]

        return classification

    def predict(self, x_new: np.ndarray) -> np.ndarray:
        """
        Predict the classification of each new point using K nearest neighbors method.
        If two or more classes receive the same number of votes, the smallest label
        is chosen.

        Args:
            x_new (np.ndarray): New points to classify with shape (n_new_samples, n_features)

        Returns:
            np.ndarray: New array of classifications with shape (n_new_samples)
        """

        if (self._x_train is None) or (self._y_train is None):
            raise ValueError("Cannot predict before training. Use '.fit()' first")

        # Convert lists and DataFrames to NumPy arrays (iterate over rows, not columns)
        x_new = np.asarray(x_new)

        predictions = [self._predict_point(sample) for sample in x_new]

        return np.array(predictions)

    def change_K(self, k_new) -> None:
        """
        Change the hyperparameter K and re-train the model with the new K

        Args:
            k_new (int): new value of K
        """

        # K validation: K negative or not int parameter
        if not isinstance(k_new, (int, np.integer)) and not (
            isinstance(k_new, float) and k_new.is_integer()
        ):
            raise TypeError("K must be an integer")
        elif k_new < 1:
            raise ValueError("K must be an integer greater than 0")

        # If already trained, the model needs at least K points
        if self._x_train is not None and len(self._x_train) < k_new:
            raise ValueError(
                f"KNN model needs at least K = {k_new} points to work ({len(self._x_train)} passed)"
            )

        self._k = int(k_new)

        # Re-train only if the model was already trained
        if self._x_train is not None:
            self.fit(self._x_train, self._y_train)
