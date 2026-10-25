import numpy as np

from ml_library.models.neural_network.layers import Dense

def test_dense_forward():
    layer = Dense(2, 1)
    layer.weights = np.array([[2.0], [3.0]])
    layer.bias = np.array([[1.0]])

    inputs = np.array([[4.0,  5.0]])
    output = layer.forward(inputs)

    np.testing.assert_allclose(output, [[24.0]])

