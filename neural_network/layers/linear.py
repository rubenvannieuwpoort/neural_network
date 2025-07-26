import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        self.x = None
        self.param = {
            'W': np.random.randn(in_features, out_features) * (2 / in_features)**.5,
            'b': np.zeros((1, out_features)),
        }
        self.grad = { p: np.zeros_like(self.param[p]) for p in self.param }

    def forward(self, x):
        self.x = x
        return x @ self.param['W'] + self.param['b']

    def backward(self, grad):
        self.grad['W'] += self.x.T @ grad
        self.grad['b'] += np.sum(grad, axis=0, keepdims=True)
        return grad @ self.param['W'].T
