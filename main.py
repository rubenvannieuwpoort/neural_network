from batcher import Batcher
from nn import Model, Linear, ReLU, MSELoss
from mnist import load_dataset

training_set = list(load_dataset('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'))
test_set = list(load_dataset('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'))

model = Model([
    Linear(784, 64),
    ReLU(),
    Linear(64, 10),
], MSELoss())

batcher = Batcher(training_set, 64)
for batchidx in range(0, 50000):
    batch = batcher.get_batch()
    total_loss = model.train(batch, 0.01)

    if batchidx % 50 == 0:
        print(f'{batchidx}; {total_loss}')
