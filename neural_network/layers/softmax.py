import numpy as np


class Softmax:
    def __init__(self):
        self.param = {}
        self.x = None

    def forward(self, x):
        # compute e**(x - x_max) for every element of x, where x_max is the largest element of x
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.y = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.y

    def backward(self, grad):
        # This is completely vibe-coded because my original implementation was wrong
        # I need to have another look at softmax and derive the backward pass myself 

        batch_size, num_classes = self.y.shape
        grad_input = np.zeros_like(self.y)
        
        for i in range(batch_size):
            y_i = self.y[i].reshape(-1, 1)
            jacobian = np.diagflat(y_i) - np.dot(y_i, y_i.T)
            grad_input[i] = jacobian.dot(grad[i])
            
        return grad_input
