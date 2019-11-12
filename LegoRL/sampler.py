from .RLmodule import *

class Sampler(RLmodule):
    """
    Basic uniform mini-batch sampling from replay buffer.
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        replay - RLmodule with "buffer", "__len__" and "n_steps" properties
        batch_size - size of sampled batches, int
        cold_start - size of buffer before providing samples, int

    Provides: sample
    """
    def __init__(self, replay, batch_size=32, cold_start=100, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert cold_start >= batch_size, "Batch size must be smaller than cold_start!"        

        self.replay = Reference(replay)
        self.batch_size = batch_size
        self.cold_start = cold_start

        self._sample = None
    
    @property
    def sample(self):
        return self._sample

    def wait(self):
        self._sample = None

    def generate_sample(self):
        """
        Samples a mini-batch.
        """
        self.debug("samples new batch uniformly.")
        sample = zip(*random.sample(self.replay.buffer, self.batch_size))
        self._sample = self.system.Batch(*sample, self.replay.n_steps)
    
    def iteration(self):
        """
        Checks if cold start condition is satisfied and samples a mini-batch.
        """
        if len(self.replay) < self.cold_start:
            self._sample = None
            self.debug("Cold Start regime: batch is not sampled.")
        else:
            self.generate_sample()

    def __repr__(self):
        return f"Each {self.timer} iteration samples mini-batch from {self.replay.name}"