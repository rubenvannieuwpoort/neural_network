from neural_network import Model
from neural_network.trainer import Trainer
from neural_network.optimizers import NoOptimizer
from neural_network.layers import Linear, ReLU
from neural_network.loss_functions import MeanSquareError
from neural_network.helpers.mnist import load_dataset

import numpy as np
import sys

from neural_network.layers import Linear, ReLU
from neural_network.loss_functions import MeanSquareError

# load training and test datasets
training_set = list(load_dataset('data/mnist/train-images-idx3-ubyte', 'data/mnist/train-labels-idx1-ubyte'))
test_set = list(load_dataset('data/mnist/t10k-images-idx3-ubyte', 'data/mnist/t10k-labels-idx1-ubyte'))

# define the model
model = Model([
    Linear(784, 64),
    ReLU(),
    Linear(64, 10),
], MeanSquareError())


def accuracy(model, dataset):
    return sum(1 for x, y_ref in dataset if np.argmax(model.forward(x)) == np.argmax(y_ref)) / len(dataset)

def print_accuracy(i):
    # print accuracies both to stdout and stderr
    tracc, teacc = accuracy(model, training_set), accuracy(model, test_set)
    print(f'{i};{tracc:.3f};{teacc:.3f}')
    sys.stderr.write(f'{i};{tracc:.3f};{teacc:.3f}\n')
    sys.stderr.flush()

# train in minibatches of size 32, don't use an optimizer
trainer = Trainer(model, training_set, 32, 1, NoOptimizer())


print('iterations;training acc;test acc')

# train in 50 iterations of 1000 batches with learning rate 0.001
for i in range(50):
    print_accuracy(i)
    trainer.train(1000, 0.001)

print_accuracy(50)
