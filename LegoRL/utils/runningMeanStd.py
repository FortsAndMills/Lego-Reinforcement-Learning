import numpy as np

"""
Standard normalization code
Adopted from OpenAI Baselines
"""

class RunningMeanStd:
    """
    Computes running mean and variance.

    Args:
      shape - tuple, shape of the statistics.
      eps - float, a small constant used to prevent variance from being 0
    """
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

    def __init__(self, shape=(), clip=None, eps=1e-8):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.clip = clip
        self.count = 0
        self.eps = eps

    def update(self, batch):
        """ Updates the running statistics given a batch of samples. """
        assert batch.shape[1:] == self.mean.shape, \
            f"batch has invalid shape: {batch.shape}, expected shape {(None,) + self.mean.shape}"

        delta = np.mean(batch, axis=0) - self.mean
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        tot_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / tot_count
        self.var = (
            self.var * (self.count / tot_count)
            + batch_var * (batch_count / tot_count)
            + np.square(delta) * (self.count * batch_count / tot_count ** 2))
        self.count = tot_count

    def apply(self, batch, center=True):
        processed = (batch - center * self.mean) / np.sqrt(self.var + self.eps)
        if self.clip is None:
            return processed
        return np.clip(processed, -self.clip, self.clip)