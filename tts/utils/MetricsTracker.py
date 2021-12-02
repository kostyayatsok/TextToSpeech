
from collections import defaultdict

class MetricsTracker:
    def __init__(self, keys):
        self.n = defaultdict(int)
        self.values = defaultdict(int)
        self.keys = keys
    def __call__(self, batch):
        for key in self.keys:
            self.values[key] += batch[key]
            self.n[key] += 1
    def __getitem__(self, key):
        if key not in self.n or self.n[key] == 0:
            return 1e9
        avg = self.values[key] / self.n[key]
        self.values[key], self.n[key] = 0, 0
        return avg
