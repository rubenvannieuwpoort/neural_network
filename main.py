from nn import Model, Linear, ReLU, MSELoss
from helpers.batcher import Batcher
from helpers.mnist import load_dataset
import numpy as np
import sys


def accuracy(model, dataset):
    correct = 0
    total = len(dataset)
    
    for x, y_ref in dataset:
        y_pred = model.forward(x)
        predicted_class = np.argmax(y_pred)
        actual_class = np.argmax(y_ref)

        if predicted_class == actual_class:
            correct += 1

    return correct / total


training_set = list(load_dataset('data/mnist/train-images-idx3-ubyte', 'data/mnist/train-labels-idx1-ubyte'))
test_set = list(load_dataset('data/mnist/t10k-images-idx3-ubyte', 'data/mnist/t10k-labels-idx1-ubyte'))

model = Model([
    Linear(784, 64),
    ReLU(),
    Linear(64, 10),
], MSELoss())

batch_size = 64
batcher = Batcher(training_set, batch_size)

print(f'batch; train accuracy; test accuracy')

num_batches = 1000
learning_rate = 0.001 / batch_size
for batchidx in range(0, num_batches):
    batch = batcher.get_batch()
    
    for x, y_ref in batch:
        y = model.forward(x)
        model.loss_function.forward(y, y_ref)
        model.backward(model.loss_function.backward())

    for layer in model.layers:
        for p in layer.param:
            layer.param[p] -= learning_rate * layer.grad[p]

    for layer in model.layers:
        for p in layer.param:
            layer.grad[p].fill(0)

    if batchidx % 100 == 0:
        sys.stderr.write(f'{batchidx / num_batches * 100:.2f}%\n')
        sys.stderr.flush()
        print(f'{batchidx}; {accuracy(model, training_set)}; {accuracy(model, test_set)}')
