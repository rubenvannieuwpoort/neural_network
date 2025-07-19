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
        
        # Extract inputs and outputs
        inputs = [sample[0] for sample in batch_samples]
        outputs = [sample[1] for sample in batch_samples]
        
        # Stack into matrices
        input_matrix = np.vstack(inputs)  # Shape: (batch_size, 784)
        output_matrix = np.vstack(outputs)  # Shape: (batch_size, 10)
        
        return input_matrix, output_matrix
