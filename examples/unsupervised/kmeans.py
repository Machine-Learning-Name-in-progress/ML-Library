import numpy as np

from ml_library import Kmeans
from ml_library.utils._random import get_rng

rng = get_rng(67)

if __name__ == "__main__":
    x_1 = rng.normal(loc=[5, 3], size=[100, 2])
    x_2 = rng.normal(loc=[-2, 5], size=[80, 2])
    x_3 = rng.normal(loc=[0, 0], size=[40, 2])

    X = np.concatenate([x_1, x_2, x_3])
    km = Kmeans(3, rng)
    km.fit(X)

    print(km.centroids)
    print(km.clusters)

    try:
        import matplotlib.pyplot as plt  # Required to view graph

        fig, ax = plt.subplots(figsize=(7, 6))

        scatter = ax.scatter(
            X[:, 0], X[:, 1], c=km.clusters, cmap="viridis", s=30, alpha=0.8
        )
        ax.scatter(
            km.centroids[:, 0],
            km.centroids[:, 1],
            c="black",
            marker="X", #type: ignore
            s=200,
            edgecolors="white",
            linewidths=1.5,
            label="Centroids",
        )

        ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper left")
        ax.legend(loc="lower right")

        ax.set_title(f"K-Means clustering (k={km.k})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
    except ModuleNotFoundError:
        print("Matplotlib is not installed")
