from LegoRL.core.RLmodule import RLmodule

from numpy.random import randint

class Sampler(RLmodule):
    """
    Basic uniform mini-batch sampling from replay buffer.
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        replay - ReplayBuffer
        batch_size - size of sampled batches, int
        cold_start - size of buffer before providing samples, int

    Provides: sample
    """
    def __init__(self, par, replay, batch_size=32, cold_start=100):
        super().__init__(par)

        assert cold_start >= batch_size, "Batch size must be smaller than cold_start!"
        self.replay = replay
        self.batch_size = batch_size
        self.cold_start = cold_start

    def sample(self):
        """
        Generates a new mini-batch into self._sample.
        output: Storage
        """
        if len(self.replay) >= self.cold_start:
            indices = randint(0, len(self.replay), self.batch_size)
            return self.replay.at(indices)

    def hyperparameters(self):
        return {"batch_size": self.batch_size, "cold_start": self.cold_start}

    def __repr__(self):
        return f"Samples mini-batches uniformly"