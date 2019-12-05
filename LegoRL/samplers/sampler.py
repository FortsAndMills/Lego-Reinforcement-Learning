from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.composed import Reference
from LegoRL.buffers.batch import Batch

import random

class Sampler(RLmodule):
    """
    Basic uniform mini-batch sampling from replay buffer.
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        replay - RLmodule with "buffer" and "__len__" properties
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

    def _generate_sample(self):
        """
        Generates a new mini-batch into self._sample.
        output: Batch
        """
        self.debug("samples new batch uniformly.")
        transitions = random.sample(self.replay.buffer, self.batch_size)
        self._sample = Batch.from_list(transitions).to_torch(self.system)
        return self._sample

    def sample(self):
        """
        Checks if cold start condition is satisfied and samples a mini-batch.
        output: Batch
        """
        if self.performed:
            self.debug("returns same sample.")
            return self._sample
        self.performed = True

        if len(self.replay) < self.cold_start:
            self._sample = None
            self.debug("cold start regime: batch is not sampled.")
        else:
            return self._generate_sample()

    def __repr__(self):
        return f"Samples mini-batches from <{self.replay.name}>"