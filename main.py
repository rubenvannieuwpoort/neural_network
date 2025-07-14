from batcher import Batcher
from nn import Model, Linear, ReLU, MSELoss
from mnist import load_dataset
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


training_set = list(load_dataset('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'))[0:7500]
test_set = list(load_dataset('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'))

model = Model([
    Linear(784, 64),
    ReLU(),
    Linear(64, 10),
], MSELoss())

batcher = Batcher(training_set, 64)

print(f'batch; train accuracy; test accuracy')

num_batches = 250000
for batchidx in range(0, num_batches):
    batch = batcher.get_batch()
    total_loss = model.train(batch, 0.025)

    if batchidx % 100 == 0:
        sys.stderr.write(f'{batchidx / num_batches * 100:.2f}%\n')
        sys.stderr.flush()
        print(f'{batchidx}; {accuracy(model, training_set)}; {accuracy(model, test_set)}')
