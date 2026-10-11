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
            raise ValueError("K must be atleast 1")
        
        self.seed = seed
        self.max_iter = max_iter

        self._centroids = None
        self._clusters = None

    @property
    def centroids(self) -> np.ndarray:
        if self._centroids is None:
            raise RuntimeError("Model is not yet fitted, call fit() first.")
        return self._centroids

    @property
    def clusters(self) -> np.ndarray:
        if self._clusters is None:
            raise RuntimeError("Model is not yet fitted, call fit() first.")
        return self._clusters

    # todo: Move this verification to a library-wide _verification Module or something as it's useful for more cases.
    def _check1DArray(self, points: np.ndarray):
        if points.ndim != 2:
            raise ValueError(
                "Expected a 2-D array of shape (n_points, n_features). "
                "Use x.reshape(1, -1) for a single point or x.reshape(-1, 1) for 1-D data."
            )
                
    def fit(self, points: np.ndarray) -> "Kmeans":
        """
        Runs the k-means algorithm on a set of points.

        Sets self.centroids and self.clusters

        Args:

            points(np.ndarray): A collection of points that will be separated in k-clusters

        Returns:

            Self
        """

        self._check1DArray(points)

        rng = _random.get_rng(self.seed)
        
        n = points.shape[0]

        if self.k > n:
            raise ValueError("K means cannot be run with k greater than the ammount of points.")
        elif self.k == n:
            self._centroids = points.copy()
            self._clusters = np.arange(n)
            return self
        elif self.k == 1:
            self._centroids = np.array([points.mean(axis=0)])
            self._clusters = np.zeros(n, dtype=int)
            return self
        
        self._centroids = self._initializeCentroids(self.k, points, rng)
        self._clusters = None

        # We repeat the search for new centroids until the clusters no longer change in comparison to the last iteration.

        for _ in range(self.max_iter):
            newClusters = self._findNearestCentroid(self._centroids, points)

            if self._clusters is not None and np.array_equal(newClusters, self._clusters):
                break

            self._clusters = newClusters
            self._centroids = self._findNewCentroids(newClusters, points, self._centroids)
        else:
            warnings.warn("Max-iteration reached, K-means did not achieve convergence")

        return self
    
    def predict(self, new_points: np.ndarray) -> np.ndarray:
        """
        Returns cluster prediction for a set of points.

        Args:
            new_points(np.ndarray): A collection of points that will be matched to their closest cluster.

        Returns:
            np.ndarray
        """

        self._check1DArray(new_points)

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

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    #_random.set_seed(3)

    data_rng = _random.get_rng(10)
    X = data_rng.normal(size=(60, 2))

    #print(X)

    kmeans = Kmeans(4, seed=10)
    kmeans.fit(X)

    H = kmeans._centroids
    c = kmeans._clusters

    


    def plot_clusters(X, labels, centroids, title=None):
        k = len(centroids)
        color_args = dict(cmap="tab10", vmin=0, vmax=9)

        plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, alpha=0.7, **color_args) #type:ignore
        plt.scatter(centroids[:, 0], centroids[:, 1], c=np.arange(k),
                    marker="X", s=200, edgecolors="black", linewidths=1.5, #type:ignore
                    **color_args) #type:ignore

        plt.gca().set_aspect("equal")
        if title:
            plt.title(title)
        plt.show()

    plot_clusters(X, c, H)
    
    

        
