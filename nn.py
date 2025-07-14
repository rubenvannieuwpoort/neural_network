import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01  # TODO: Kaiming initialization
        self.b = np.zeros((1, out_features))
        self.x, self.dW, self.dB = None, None, None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = (grad.T * self.x).T
        self.db = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T


class ReLU:
    def __init__(self):
        self.x = None
    
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)
    
    def backward(self, grad):
        return grad * (self.x > 0)


class MSELoss:
    def forward(self, y, y_ref):
        self.y = y
        self.y_ref = y_ref
        return np.mean((y - y_ref) ** 2)
    
    def backward(self):
        return 2 * (self.y - self.y_ref) / self.y.shape[1]


class Model:
    def __init__(self, layers, loss_function):
        self.layers = layers
        self.loss_function = loss_function

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def update(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, 'W'):
                layer.W -= learning_rate * layer.dW
                layer.b -= learning_rate * layer.db

    def train(self, batch, learning_rate):
        total_loss = 0
        for x, y_ref in batch:
            y = self.forward(x)
            total_loss += self.loss_function.forward(y, y_ref)
            self.backward(self.loss_function.backward())

        self.update(learning_rate / len(batch))
        return total_loss / len(batch)

    def loss(self, batch):
        loss = 0
        for x, y_ref in batch:
            loss += self.loss_function.forward(self.forward(x), y_ref)

        return loss / len(batch)
