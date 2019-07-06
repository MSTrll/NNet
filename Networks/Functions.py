import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def J_quadratic(neuron, X, y):
    assert y.shape[1] == 1, 'Incorrect y shape'
    return 0.5 * np.mean((neuron.vectorized_forward_pass(X) - y) ** 2)


def J_quadratic_derivative(y, y_hat):
    assert y_hat.shape == y.shape and y_hat.shape[1] == 1, 'Incorrect shapes'
    return (y_hat - y) / len(y)


def cost_function(network, test_data, onehot=True):
    c = 0
    for example, y in test_data:
        if not onehot:
            y = np.eye(3, 1, k=-int(y))
        yhat = network.feedforward(example)
        c += np.sum((y - yhat) ** 2)
    return c / len(test_data)
