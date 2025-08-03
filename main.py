import numpy as np

from neural_network import Model
from neural_network.trainer import Trainer
from neural_network.optimizers import NoOptimizer
from neural_network.layers import Linear, ReLU
from neural_network.loss_functions import MeanSquareError
from neural_network.dataloaders import MNIST


# load training and test datasets
training_set, test_set = MNIST.training_set(), MNIST.test_set()


def one_hot(k):
    return np.array([1 if i == k else 0 for i in range(10)])

# convert labels to one-hot representation
training_set = list(map(lambda x: (x[0], one_hot(x[1])), training_set))
test_set = list(map(lambda x: (x[0], one_hot(x[1])), test_set))


# define the model
model = Model([
    Linear(784, 64),
    ReLU(),
    Linear(64, 10),
], MeanSquareError())


def accuracy(model, dataset):
    return sum(1 for x, y_ref in dataset if np.argmax(model.forward(x)) == np.argmax(y_ref)) / len(dataset)

def print_accuracy(i):
    tracc, teacc = accuracy(model, training_set), accuracy(model, test_set)
    print(f'{i};{tracc:.3f};{teacc:.3f}')

# train in minibatches of size 32, don't use an optimizer
trainer = Trainer(model, training_set, 32, 1, NoOptimizer())

print('iterations;training acc;test acc')

for i in range(25):
    print_accuracy(i)
    trainer.train(100, 1)

for i in range(25, 50):
    print_accuracy(i)
    trainer.train(100, 0.1)

print_accuracy(50)
