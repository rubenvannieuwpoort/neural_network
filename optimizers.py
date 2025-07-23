class NoOptimizer:
    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

    def step(self, learning_rate, samples_per_step):
        learning_rate /= samples_per_step

        for layer in self.model.layers:
            for p in layer.param:
                layer.param[p] -= learning_rate * layer.grad[p]
