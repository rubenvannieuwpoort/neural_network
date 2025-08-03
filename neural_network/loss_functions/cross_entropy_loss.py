import numpy as np


class CrossEntropyLoss:
    def forward(self, y, y_ref):
        self.y = y
        self.y_ref = y_ref
        return -np.sum(y_ref * np.log(self.y))  / y_ref.shape[0]

    def backward(self):
        return -(self.y_ref / self.y) / self.y_ref.shape[0]
