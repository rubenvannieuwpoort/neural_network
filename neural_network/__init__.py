import pickle
import numpy as np


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

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
