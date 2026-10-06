import numpy as np
from ml_library.utils import _random
import warnings

class Kmeans:
    """
    K-Means++ unsupervised clustering method.
    
    Args:
        k (int): Number of centroids
        X (np.ndarray): Data points

    """
    def __init__(self,k: int, seed = None, max_iter: int = 300) -> None:
        self.k = k

        if self.k < 1:
            raise ValueError("K cannot be lesser than 1")
        
        self.seed = seed
        self.max_iter = max_iter

        self.centroids = None
        self.clusters = None
            
    def fit(self, points: np.ndarray) -> None:
        """
        Runs the k-means algorithm on a set of points.

        Sets self.centroids and self.clusters

        Args:

            points(np.ndarray): A collection of points that will be separated in k-clusters

        Returns:

            None
        """

        rng = _random.get_rng(self.seed)
        
        n = points.shape[0]

        if self.k > n:
            raise ValueError("K means cannot be run on less than k points.")
        elif self.k == n:
            self.centroids = points.copy()
            self.clusters = np.arange(n)
            return
        elif self.k == 1:
            self.centroids = np.array([points.mean(axis=0)])
            self.clusters = np.zeros(n, dtype=int)
            return
        
        self.centroids = self._initializeCentroids(self.k, points, rng)
        self.clusters = None

        # We repeat the search for new centroids until the clusters no longer change in comparison to the last iteration.

        for _ in range(self.max_iter):
            newClusters = self._findNearestCentroid(self.centroids, points)

            if self.clusters is not None and np.array_equal(newClusters, self.clusters):
                break

            self.clusters = newClusters
            self.centroids = self._findNewCentroids(newClusters, points, self.centroids)
        else:
            warnings.warn("Max-iteration reached, K-means did not achieve convergence")
    
    def predict(self, new_points: np.ndarray) -> np.ndarray:
        """
        Returns cluster prediction for a set of points.

        Args:
            new_points(np.ndarray): A collection of points that will be matched to their closest cluster.

        Returns:
            np.ndarray
        """

        if self.centroids is None:
            raise RuntimeError("Model is not yet fitted, call fit() first.")
        
        if new_points.shape[1] != self.centroids.shape[1]:
            raise ValueError("Dimension of new_points doesn't match the existing ones.")

        return self._findNearestCentroid(self.centroids, new_points)

    def _findNewCentroids(self, labels, points, old_centroids):
        centroids = np.empty((old_centroids.shape[0], points.shape[1]))

        for i in range(old_centroids.shape[0]):
            mask = labels == i # True if J-th point belongs to category i, false otherwise

            if mask.any():
                centroids[i] = points[mask].mean(axis=0)
            else:
                centroids[i] = old_centroids[i]

        return centroids
        


    def _findNearestCentroid(self,centroids, points):
        # We want ||x - c||^2 which is the squared distance to each centroid

        # We can factor it as ||x||^2 - 2xc + ||c||^2

        # Note that we can drop the ||x||^2 as it is the same number applied to each centroid, so we can just look for smallest number that (||c||^2 - 2xc) gives

        # x_sq = np.sum(points**2, axis=1, keepdims=True)
        c_sq = np.sum(centroids**2, axis=1)

        xc = points @ centroids.T

        # We pick the maximum between the distance and zero, because for points very close to the centroids, we can have rounding errors and get negative distances
        # distances_sq = np.maximum(x_sq - 2*xc + c_sq, 0) 

        scores = c_sq - 2*xc
        categories = np.argmin(scores, axis=1)

        return categories

    def _initializeCentroids(self,k,points,rng):
        """
        Uses the k-means++ initial centroid strategy so the starting point is closer to convergence, and it's also better than falling in to local minimum solutions.
        """

        n = points.shape[0]
        centroids = np.empty((k, points.shape[1]))

        centroids[0] = points[rng.integers(n)]
        dsquare = ((points - centroids[0]) ** 2).sum(axis=1)
        

        for i in range(1, k):
            dsqsum = dsquare.sum()

            if dsqsum == 0:
                centroids[i] = points[rng.choice(n)]
            else:
                centroids[i] = points[rng.choice(n, p=dsquare/dsqsum)] # The bigger the distance, the larger the probability of being chosen.

            dsquare = np.minimum(dsquare, ((points - centroids[i])**2).sum(axis=1))

        return centroids
    

        
