class EarlyStopping:
    def __init__(self, patience=1000, min_delta=0.0, eval_every=100):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.count = 0
        self.eval_every = eval_every

    def step(self, value):
        if value < self.best - self.min_delta:
            self.best = value
            self.count = 0
        else:
            self.count += 1
        return self.count >= self.patience / self.eval_every
