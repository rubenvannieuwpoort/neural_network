import numpy as np


class MeanSquareError:
    def forward(self, y, y_ref):
        self.y = y
        self.y_ref = y_ref
        return np.mean((y - y_ref) ** 2)

    def backward(self):
        return (2 / self.y.shape[1]) * (self.y - self.y_ref)
