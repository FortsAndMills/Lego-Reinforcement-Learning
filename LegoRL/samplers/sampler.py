from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference
from LegoRL.buffers.storage import Storage

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
    def __init__(self, replay, batch_size=32, cold_start=100, timer=1, frozen=False):
        super().__init__(timer=timer, frozen=frozen)

        assert cold_start >= batch_size, "Batch size must be smaller than cold_start!"        

        self.replay = Reference(replay)
        self.batch_size = batch_size
        self.cold_start = cold_start

        self._sample = None

    def _generate_sample(self):
        """
        Generates a new mini-batch into self._sample.
        output: Storage
        """
        self.debug("samples new batch uniformly.")
        transitions = random.sample(self.replay.buffer, self.batch_size)
        self._sample = self.mdp[Storage].from_list(transitions)
        return self._sample

    def sample(self):
        """
        Checks if cold start condition is satisfied and samples a mini-batch.
        output: Storage
        """
        if self._performed:
            self.debug("returns same sample.")
            return self._sample
        self._performed = True

        if len(self.replay) < self.cold_start:
            self._sample = None
            self.debug("cold start regime: batch is not sampled.")
        else:
            return self._generate_sample()

    def hyperparameters(self):
        return {"batch_size": self.batch_size, "cold_start": self.cold_start}

    def __repr__(self):
        return f"Samples mini-batches from <{self.replay.name}>"