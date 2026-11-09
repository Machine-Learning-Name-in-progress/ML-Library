"""decision tree regressor using MSE impurity."""
from .tree import decision_tree

class decision_tree_regressor (decision_tree):
    """regression on samples using a decision tree

    Args:
        max_depth (int): max deapth of the tree, default = 100
    """

    def __init__(self, max_deapth: int = 100, min_samples_split: int = 1, min_samples_leaf: int = 1) -> None:
        super().__init__(max_deapth, 2, min_samples_split, min_samples_leaf)
