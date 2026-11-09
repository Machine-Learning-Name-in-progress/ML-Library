import numpy as np

from ml_library import LDA
from ml_library.metrics import accuracy, confusion_matrix
from ml_library.utils._random import get_rng

#AI use disclaimer: The plot visualization was made by claude-opus-5.5

rng = get_rng(67)
test_size = 0.2

if __name__ == "__main__":
    x_1 = rng.normal(loc=[5, 3], size=[100, 2])
    x_2 = rng.normal(loc=[-2, 5], size=[80, 2])
    x_3 = rng.normal(loc=[0, 0], size=[40, 2])

    X = np.concatenate([x_1, x_2, x_3])
    y = np.array(["dog"] * 100 + ["cat"] * 80 + ["bird"] * 40)

    # Shuffle the indices so X and y stay aligned, then split into train/test
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    lda = LDA()
    lda.fit(X_train, y_train)

    y_pred = lda.predict(X_test)
    acc = accuracy(y_test.tolist(), y_pred.tolist())

    print(f"Classes: {lda.classes}")
    print(f"Priors:  {lda.priors}")
    print(f"Test accuracy: {acc:.3f}")
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_test.tolist(), y_pred.tolist(), lda.classes.tolist()))

    try:
        import matplotlib.pyplot as plt  # Required to view graph

        # Classify every point of a grid to draw the decision regions
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300)
        )
        grid = np.column_stack([xx.ravel(), yy.ravel()])

        # Map string labels to integers so matplotlib can color them
        class_to_int = {c: i for i, c in enumerate(lda.classes)}
        zz = np.array([class_to_int[c] for c in lda.predict(grid)]).reshape(xx.shape)

        cmap = plt.get_cmap("viridis", len(lda.classes))

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.contourf(xx, yy, zz, alpha=0.2, cmap=cmap, levels=len(lda.classes) - 1)

        for c, i in class_to_int.items():
            color = cmap(i)
            ax.scatter(*X_train[y_train == c].T, color=color, s=30, alpha=0.8, label=c)
            ax.scatter(
                *X_test[y_test == c].T, color=color, s=60, marker="^",  # type: ignore
                edgecolors="black",
            )

        ax.scatter([], [], color="gray", marker="^", edgecolors="black", label="test")  # type: ignore
        ax.legend(title="Class", loc="upper right")

        ax.set_title(f"LDA decision regions (test accuracy = {acc:.2f})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
    except ModuleNotFoundError:
        print("Matplotlib is not installed")
