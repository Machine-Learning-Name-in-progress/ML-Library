import numpy as np

# TODO: Functions organization

class knn:
    """
    K-Nearest Neighbors (KNN) classification method
    
    Args:
        k (int): Number of neighbors considered for prediction
    """
    
    def __init__(self, k:int) -> None:

        # K validation: K negative of not int parameter
        if k < 1:
            raise ValueError("K must be a integer bigger than 0")
        elif not isinstance(k, int) and not (isinstance(k, float) and k.is_integer()):
            raise TypeError("K must be a integer")
        
        self._k = k              # Hyperparameter K
        self._x_train = None    # Can be set using train()
        self._y_train = None    # Can be set using train()

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using the training datasets
        
        Args:
            x (np.ndarray): Training features matrix with shape (n_samples, n_features)
            y (np.ndarray): Classifications vector with shape (n_samples)
        """
        
        # Number of points validation
        if len(x) < self._k:
            raise ValueError(f"KNN model need atleast K = {self._k} points to work ({len(x)} passed)")
        
        self._x_train = x      # Preditors
        self._y_train = y      # Categories

    def predict(self, x_new: np.ndarray) -> str:
        """
        Predict a classification for a new data point
        
        Args:
            x_new (np.ndarray): New point to classify with shape (1, n_feature)
            
        Returns:
            str: Predicted classification for the new point
        """
        
        if len(x_new) > 1:
            return self.predictions(x_new)
        
        # Compute k-neares distances
        distances = [np.linalg.norm(x_new - x_train) for x_train in self._x_train]

        # Pick the k-nearest points (index of k-shortest distances)
        nearest = np.argsort(distances)[:self._k]

        # Captures the classification of the nearest points
        nearest_categories = self._y_treino[nearest]
        
        # The classes and counts of the nearest points
        classes, counts = np.unique(nearest_categories, return_counts=True)

        # Choose the most frequently classification
        biggest_index = np.argmax(counts)
        classification = classes[biggest_index]
        
        # TODO: Tie-breaker method (ask Rodrigo first)

        return classification
    
    def predictions(self, x_new: np.ndarray) -> np.ndarray:
        """
        Predict the classifcation for many new points
        
        Args:
            x_new (np.adarray): New point to classify with shape (n_new_samples, n_feature)
            
        Returns:
            np.ndarray: New array of classifications with shape (n_new_samples)
        """
        
        predictions = [self.predict(point) for point in x_new]
        
        return predictions
    
    # TODO: a K-changer method, if it make sense