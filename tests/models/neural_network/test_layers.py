import numpy as np

from ml_library.models.neural_network.layers import Dense

def test_dense_forward():
    layer = Dense(2, 1)
    layer.weights = np.array([[2.0], [3.0]])
    layer.bias = np.array([[1.0]])

    inputs = np.array([[4.0,  5.0]])
    output = layer.forward(inputs)

    np.testing.assert_allclose(output, [[24.0]])

def test_dense_backward():
    layer = Dense(3, 2)
    layer.weights = np.array([[1.0, 2.0], [-1.0, 3.0], [4.0, -2.0]])
    layer.bias = np.array([[0.5, -0.5]])

    inputs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    output_gradient = np.array([[2.0, -1.0], [3.0, 4.0]])
    layer.forward(inputs)
    input_gradient = layer.backward(output_gradient)

    np.testing.assert_allclose(
        layer.grad_weights, [[14.0, 15.0], [19.0, 18.0], [24.0, 21.0]]
    )
    
    np.testing.assert_allclose(layer.grad_bias, [[5.0, 3.0]])
    np.testing.assert_allclose(input_gradient, [[0.0, -5.0, 10.0], [11.0, 9.0, 4.0]])