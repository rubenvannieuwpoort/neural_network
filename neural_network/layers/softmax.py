import numpy as np


class Softmax:
    def __init__(self):
        self.param = {}
        self.x = None

    def forward(self, x):
        self.y = np.exp(x) / np.sum(np.exp(x))
        return self.y

    def backward(self, grad):
        return grad * self.y * (1 - self.y)
