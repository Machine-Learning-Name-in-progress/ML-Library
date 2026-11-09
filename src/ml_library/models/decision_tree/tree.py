from typing import Self

import numpy as np

from ._utils import _best_split, _gini, _is_pure, _majority_choice, _mean_choice, _mse

class _decision_tree_node:
    left: Self | None = None
    right: Self| None = None

    prediction = None
    target_X: int | None = None

    deapth: int = 0
    threshold: float = 0

    def __init__(self, prediction, deapth: int):
        self.deapth = deapth
        self.prediction = prediction

    def predict(self, x: np.ndarray) -> np.ndarray:
        y = np.full(x.shape[0], self.prediction, dtype=object)

        if self.target_X is None:
            return y

        left = x[:, self.target_X] <= self.threshold
        
        if self.left:
            y[left] = self.left.predict(x[left])
        if self.right:
            y[~left] = self.right.predict(x[~left])

        return y



def _Alocate_node(impurity, impurity_choice, x: np.ndarray, y: np.ndarray, deapth: int, max_deapth: int = 10, min_samples_split: int = 1, min_samples_leaf: int = 1) -> _decision_tree_node:
    prediction = impurity_choice(y)

    node = _decision_tree_node(prediction, deapth)

    if deapth >= max_deapth or _is_pure(y) or min_samples_split > np.size(y):
        return node

    best_split = _best_split(x,y,impurity, min_samples_leaf)
    if best_split is None:
        return node

    target_X, threshold = best_split
    node.target_X = target_X
    node.threshold = threshold

    left = x[:, target_X] <= threshold

    node.left = _Alocate_node(impurity,     impurity_choice, x[left],   y[left],    deapth+1, max_deapth, min_samples_split, min_samples_leaf)
    node.right = _Alocate_node(impurity,    impurity_choice, x[~left],  y[~left],   deapth+1, max_deapth, min_samples_split, min_samples_leaf)
    
    return node




class decision_tree:
    _root : _decision_tree_node | None = None
    _max_deaph: int = 100
    _impurity = None
    _impurity_choice = None
    _min_samples_split: int = 1
    _min_samples_leaf: int = 1

    def __init__(self, max_deapth: int = 100, impurity_type: int = 1, min_samples_split: int = 1, min_samples_leaf: int = 1):
        '''
            Args:
                impurity_type (int): 1 for GINI and 2 for MSE
        '''
        self._max_deaph = max_deapth
        self._min_samples_leaf = min_samples_leaf
        self._min_samples_split = min_samples_split

        if impurity_type == 1:
            self._impurity = _gini
            self._impurity_choice = _majority_choice

        elif impurity_type == 2:
            self._impurity = _mse
            self._impurity_choice = _mean_choice

        else: raise ValueError(f"impurity_type={impurity_type} isn't a default option")

    def fit(self, x: np.ndarray, y: np.ndarray):
        self._root = _Alocate_node(self._impurity, self._impurity_choice, x, y, 0, self._max_deaph, self._min_samples_split, self._min_samples_leaf)

    def predict(self, x:np.ndarray) -> np.ndarray:
        if self._root is None:
            raise ValueError("[Root is not setted]: Call fit before calling predict")
        return self._root.predict(x)
