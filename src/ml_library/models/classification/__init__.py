"""Classification models."""

from .discriminant_analysis import LDA, QDA
from .knn import KNN
from .logistic_regression import LogisticRegression

__all__ = ["KNN", "LDA", "LogisticRegression", "QDA"]
