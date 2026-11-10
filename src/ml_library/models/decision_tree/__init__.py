"""Public interface for decision-tree models.

Implementation roadmap:
    tree.py: Add shared node storage, recursive growth, and traversal.
    classifier.py: Extend the current Gini stump to use recursive growth.
    regressor.py: Add numeric targets, squared-error splits, and mean leaves.
    tests/models/test_decision_tree.py: Cover predictions and stopping rules.
"""

from .classifier import decision_tree_classifier
from .regressor import decision_tree_regressor

__all__ = [
    "decision_tree_classifier",
    "decision_tree_regressor"
]
