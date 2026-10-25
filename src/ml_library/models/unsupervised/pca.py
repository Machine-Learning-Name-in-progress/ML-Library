import numpy as np

class pca:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components = None

    def fit(self, X):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def standardize(self, X):
        pass