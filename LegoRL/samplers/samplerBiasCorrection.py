from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference

import numpy as np
from copy import copy

class SamplerBiasCorrection(RLmodule):
    """
    Weighted importance sampling to correct bias in prioritized replays.
    When sample is requested, takes sample from other sampler and adds weights to compensate bias.
    Based on: https://arxiv.org/abs/1511.05952

    Args:
        sampler - RLmodule, providing "sample" 
        rp_beta_start - float, degree of importance sampling to compensate bias, from 0 to 1
        rp_beta_iterations - int, number of iterations till unbiased sampling

    Provides: sample
    """
    def __init__(self, sampler, rp_beta_start=0.4, rp_beta_iterations=100000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sampler = Reference(sampler)

        self.hyperparameters = lambda: {"rp_beta_start": rp_beta_start, "rp_beta_iterations": rp_beta_iterations}
        self.rp_beta = lambda: min(1.0, rp_beta_start + self.system.iterations * (1.0 - rp_beta_start) / rp_beta_iterations)

        self._sample = None

    def sample(self):
        '''
        Returns a sample of batches.
        output: Storage
        '''
        if self._performed:
            self.debug("returns same sample")
            return self._sample
        self._performed = True
        
        original_sample = self.sampler.sample()
        if original_sample is None:
            self._sample = None
            self.debug("no weights added as there is no sample.")
            return

        if self.frozen:
            self._sample = original_sample
            self.debug("no weights added as module is frozen.")
            return

        # we copy this batch in case someone wants to use
        # same mini-batch without bias correction
        self._sample = copy(original_sample)
        assert hasattr(self._sample, "priorities"), "Can't correct bias in a sample without priorities."
        
        # calculating importance sampling weights to evade bias
        # these weights are annealed to be more like uniform at the beginning of learning
        weights  = (self._sample.priorities) ** (-self.rp_beta())
        # these weights are normalized as proposed in the original article to make loss function scale more stable.
        weights /= self._sample.priorities.min() ** (-self.rp_beta())

        # logs
        self.log("median weight", np.median(weights), "weights")
        self.log("mean weight", np.mean(weights), "weights")
        self.debug("weights added to sampled batch")
        
        self._sample.weights = self.mdp["Weights"](weights)
        return self._sample

    def __repr__(self):
        return f"Adds weights to mini-batches from <{self.sampler.name}> to correct bias"