import random
import numpy as np


class Batcher:
    def __init__(self, data, batch_size):
        assert batch_size <= len(data)
        self.batch_size = batch_size
        self.data = data
        self.indices = list(range(0, len(data)))

    def get_batch(self):
        random.shuffle(self.indices)
        batch_samples = [self.data[i] for i in self.indices[:self.batch_size]]
        inputs, outputs = [sample[0] for sample in batch_samples], [sample[1] for sample in batch_samples]
        return np.vstack(inputs), np.vstack(outputs)


class Trainer:
    def __init__(self, model, training_set, batch_size, batches_per_step, optimizer):
        self.model = model
        self.batcher = Batcher(training_set, batch_size)
        self.batches_per_step = batches_per_step
        self.samples_per_step = batch_size * batches_per_step
        self.optimizer = optimizer
        optimizer.set_model(self.model)

    def train(self, steps, learning_rate):
        for _ in range(steps):
            for _ in range(self.batches_per_step):
                x, y_ref = self.batcher.get_batch()

                y = self.model.forward(x)
                self.model.loss_function.forward(y, y_ref)
                self.model.backward(self.model.loss_function.backward())

            self.optimizer.step(learning_rate, self.samples_per_step)

        for layer in self.model.layers:
            for p in layer.param:
                layer.grad[p].fill(0)
