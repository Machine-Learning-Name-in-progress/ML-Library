import numpy as np

from ml_library import LDA, QDA
from ml_library.metrics import accuracy, confusion_matrix
from ml_library.utils._random import get_rng

#AI use disclaimer: The plot visualization was made by claude-opus-5.5

rng = get_rng(67)
test_size = 0.2

if __name__ == "__main__":
    x_1 = rng.multivariate_normal([0, 0], [[0.4, 0], [0, 0.4]], size=120)  # tight core
    x_2 = rng.multivariate_normal([0, 0], [[9, 0], [0, 9]], size=150)  # wide halo around it
    x_3 = rng.multivariate_normal([6, 6], [[4, -3.2], [-3.2, 4]], size=100)  # stretched diagonally

    X = np.concatenate([x_1, x_2, x_3])
    y = np.array(["core"] * 120 + ["halo"] * 150 + ["stripe"] * 100)

    # Shuffle the indices so X and y stay aligned, then split into train/test
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    models = {"LDA": LDA().fit(X_train, y_train), "QDA": QDA().fit(X_train, y_train)}

    accs = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accs[name] = accuracy(y_test.tolist(), y_pred.tolist())
        print(f"{name} test accuracy: {accs[name]:.3f}")
        print(confusion_matrix(y_test.tolist(), y_pred.tolist(), model.classes.tolist()))

    try:
        import matplotlib.pyplot as plt  # Required to view graph
        from matplotlib.colors import ListedColormap

        # Classify every point of a grid to draw the decision regions
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
        grid = np.column_stack([xx.ravel(), yy.ravel()])

        classes = models["QDA"].classes
        class_to_int = {c: i for i, c in enumerate(classes)}
        colors = ["#e4572e", "#4c72b0", "#2a9d8f"]

        fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
        for ax, (name, model) in zip(axes, models.items()):
            zz = np.array([class_to_int[c] for c in model.predict(grid)]).reshape(xx.shape)
            ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5], cmap=ListedColormap(colors), alpha=0.18)
            ax.contour(xx, yy, zz, levels=[0.5, 1.5], colors="k", linewidths=1, alpha=0.6)

            for c, i in class_to_int.items():
                ax.scatter(*X_train[y_train == c].T, color=colors[i], s=18, alpha=0.7, label=c)
                ax.scatter(*X_test[y_test == c].T, color=colors[i], s=50, marker="^", edgecolors="black", linewidths=0.8)  # type: ignore

            ax.set_title(f"{name} (test accuracy = {accs[name]:.2f})")
            ax.set_xlabel("x")
            ax.grid(alpha=0.3)

        axes[0].set_ylabel("y")
        axes[1].scatter([], [], color="gray", marker="^", edgecolors="black", label="test")  # type: ignore
        axes[1].legend(title="Class", loc="upper left")
        fig.suptitle("Same data, shared covariance (LDA) vs per-class covariance (QDA)")

        plt.tight_layout()
        plt.show()
    except ModuleNotFoundError:
        print("Matplotlib is not installed")
