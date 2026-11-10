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
    decision_tree_classifier,
    decision_tree_regressor
)

__all__ = [
    "decision_tree_classifier",
    "decision_tree_regressor",
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
