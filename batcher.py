import random


class Batcher:
    def __init__(self, data, batch_size):
        assert batch_size <= len(data)
        self.batch_size = batch_size
        self.data = data
        self.indices = list(range(0, len(data)))

    def get_batch(self):
        random.shuffle(self.indices)
        return list(map(lambda i: self.data[i], self.indices[:self.batch_size]))
