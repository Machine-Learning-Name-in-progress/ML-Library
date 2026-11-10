# ML-Library

A machine learning library built from scratch with NumPy, made for the Machine Learning 1 course (MD22) at IMPA Tech.

## Installation

```bash
pip install mlnip-lib
```

Requires Python 3.11+.

## Quick example

The package is installed as `mlnip-lib` imported as `ml_library`:

```python
import numpy as np
from ml_library import OrdinaryLeastSquares
from ml_library.metrics import r_squared

X = np.array([[1, -1], [2, 5], [0, -2], [-3, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5])

model = OrdinaryLeastSquares()
model.fit(X, y)
y_pred = model.predict(X)

print(model.coef, model.intercept)
print(r_squared(y, y_pred))
```

## What's inside

| Module | Contents |
|---|---|
| `models.regression` | Ordinary Least Squares, Ridge, Lasso, Polynomial Regression |
| `models.classification` | KNN, Logistic Regression, LDA, QDA |
| `models.unsupervised` | K-Means, PCA |
| `models.neural_network` | MLP (work in progress) |
| `model_selection` | K-fold, LOOCV, bootstrap, grid search CV |
| `metrics` | Regression metrics (MSE, RMSE, MAE, MAPE, R², ...) and classification metrics (accuracy, precision, recall, F1, AUC, ...) |
| `preprocessing` | Normalization, standardization, one-hot encoding, missing value handling |

## Development

This project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Machine-Learning-Name-in-progress/ML-Library.git
cd ML-Library
uv sync
uv run pytest
```

## Documentation

[Decision Trees documentation](https://machine-learning-name-in-progress.github.io/ML-Library/)

## License

[MIT](LICENSE)
