from nn import Model, Linear, ReLU, MSELoss
from helpers.batcher import Batcher
from helpers.mnist import load_dataset
import numpy as np
import sys


def accuracy(model, dataset):
    return sum(1 for x, y_ref in dataset if np.argmax(model.forward(x)) == np.argmax(y_ref)) / len(dataset)


training_set = list(load_dataset('data/mnist/train-images-idx3-ubyte', 'data/mnist/train-labels-idx1-ubyte'))
test_set = list(load_dataset('data/mnist/t10k-images-idx3-ubyte', 'data/mnist/t10k-labels-idx1-ubyte'))

model = Model([
    Linear(784, 64),
    ReLU(),
    Linear(64, 10),
], MSELoss())

batch_size = 32
batcher = Batcher(training_set, batch_size)

print(f'batch; train accuracy; test accuracy')

num_samples = 1024*128
num_batches = num_samples // batch_size
learning_rate = 0.1 / batch_size
for batchidx in range(0, num_batches):
    x, y_ref = batcher.get_batch()

    y = model.forward(x)
    model.loss_function.forward(y, y_ref)
    model.backward(model.loss_function.backward())

    for layer in model.layers:
        for p in layer.param:
            layer.param[p] -= learning_rate * layer.grad[p]
            layer.grad[p].fill(0)

    if batchidx % 10 == 0:
        print(f'{batchidx}; {accuracy(model, training_set)}; {accuracy(model, test_set)}')
        sys.stderr.write(f'{batchidx / num_batches * 100:.2f}%\n')
        sys.stderr.flush()
