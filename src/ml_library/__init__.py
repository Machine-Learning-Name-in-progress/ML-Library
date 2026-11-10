"""ml_library — a machine learning library built from scratch."""

from . import metrics, model_selection, models, preprocessing, utils
from .models import (
    KNN,
    LDA,
    QDA,
    Kmeans,
    Lasso,
    LogisticRegression,
    OrdinaryLeastSquares,
    PolynomialRegression,
    Ridge,
    PCA,
)

__all__ = [
    "metrics",
    "model_selection",
    "models",
    "preprocessing",
    "utils",
    "KNN",
    "LDA",
    "QDA",
    "Kmeans",
    "Lasso",
    "LogisticRegression",
    "OrdinaryLeastSquares",
    "PolynomialRegression",
    "Ridge",
    "PCA",
]
