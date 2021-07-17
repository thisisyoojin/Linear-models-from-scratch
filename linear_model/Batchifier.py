import numpy as np

class Batchifier:
    
    def __init__(self):
        self.batches = []

    def batch(self, X, y, batch_size=16):
        batches = []
        while X.any():
            if len(X) <= batch_size:
                batches.append((X, y))
                break

            X_batch = np.empty((batch_size, X.shape[1]))
            y_batch = np.empty((batch_size,))

            for bdx in range(batch_size):
                num_generated = np.random.randint(0, len(X))
                X_batch[bdx] = X[num_generated]
                y_batch[bdx] = y[num_generated]
                X = np.delete(X, num_generated, axis=0)
                y = np.delete(y, num_generated, axis=0)
            batches.append((X_batch, y_batch))
        
        self.batches = batches
        return batches

    def __getitem__(self, idx):
        return self.batches[idx]

    def __len__(self):
        return len(self.batches)