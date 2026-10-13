import numpy as np

# TODO: Functions organization

class knn:
    """
    K-Nearest Neighbors (KNN) classification method
    
    Args:
        k (int): Number of neighbors considered for prediction
    """
    
    def __init__(self, k:int) -> None:

        # K validation
        if k < 1:
            raise ValueError("K must be a non-negative integer")
        
        self.k = k              # Hyperparameter K
        self._x_train = None    # Can be set using train()
        self._y_train = None    # Can be set using train()

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model with datas
        
        Args:
            x (np.ndarray)
        """
        
        # Number of points validation
        if len(x) < self.k:
            raise ValueError(f"KNN model need atleast K = {self.k} points to work ({len(x)} passed)")
        
        self._x_treino = x      # Preditors
        self._y_treino = y      # Categories

    def predict(self, x_new) -> str:
        # IMPORTANT: normalize numerical values

        # TODO: improve for more than 1 preditor
        distances = [abs(x_new - x_treino) for x_treino in self._x_treino]

        # Compute k-neares distances
        nearest = np.argsort(distances)[:self.k]

        nearest_categories = self.y_treino[nearest]
        classes, counts = np.unique(nearest_categories, return_counts=True)

        biggest_index = np.argmax(counts)
        classification = classes[biggest_index]

        return classification

    # TODO: vetorized predict function (predict many values)
    # TODO: accuracy and quality metrics (?)