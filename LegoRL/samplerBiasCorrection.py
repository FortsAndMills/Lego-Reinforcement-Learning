from .RLmodule import *
from copy import copy

class SamplerBiasCorrection(RLmodule):
    """
    Weighted importance sampling to correct bias in prioritized replay.
    Based on: https://arxiv.org/abs/1511.05952

    Args:
        sampler - RLmodule, providing "sample" 
        rp_beta_start - float, degree of importance sampling smoothing out the bias, from 0 to 1
        rp_beta_iterations - int, number of iterations till unbiased sampling
    """
    def __init__(self, sampler, rp_beta_start=0.4, rp_beta_iterations=100000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sampler = Reference(sampler)
        self.rp_beta = lambda: min(1.0, rp_beta_start + self.system.iterations * (1.0 - rp_beta_start) / rp_beta_iterations)

        self._sample = None
    
    @property
    def sample(self):
        return self._sample

    def wait(self):
        '''
        When frozen, this module leaves samples unchanged.
        '''
        self._sample = self.sampler.sample

    def iteration(self):
        '''
        Copies samples from replay and adds weights to them.
        '''
        if self.sampler.sample is None:
            self._sample = None
            self.debug("no weights added as there is no sample.")
            return

        self._sample = copy(self.sampler.sample)
        assert hasattr(self._sample, "priorities"), "Can't correct bias in a sample without priorities."
        
        # calculating importance sampling weights to evade bias
        # these weights are annealed to be more like uniform at the beginning of learning
        weights  = (self._sample.priorities) ** (-self.rp_beta())
        # these weights are normalized as proposed in the original article to make loss function scale more stable.
        weights /= self._sample.priorities.min() ** (-self.rp_beta())

        self.log("median weight", np.median(weights), "training iteration", "weights")
        self.log("mean weight", np.mean(weights), "training iteration", "weights")
    
        self._sample.weights = Tensor(weights)
        self.debug("weights added to sampled batch")

    def __repr__(self):
        return f"Adds weights to mini-batches from {self.sampler.name} to correct bias"