from LegoRL.core.RLmodule import RLmodule

import numpy as np
from copy import copy

class SamplerBiasCorrection(RLmodule):
    """
    Weighted importance sampling to correct bias in prioritized replays.
    When sample is requested, takes sample from other sampler and adds weights to compensate bias.
    Based on: https://arxiv.org/abs/1511.05952

    Args:
        rp_beta_start - float, degree of importance sampling to compensate bias, from 0 to 1
        rp_beta_iterations - int, number of iterations till unbiased sampling
    """
    def __init__(self, system, rp_beta_start=0.4, rp_beta_iterations=100000):
        super().__init__(system)

        self.hyperparameters = lambda: {"rp_beta_start": rp_beta_start, "rp_beta_iterations": rp_beta_iterations}
        self.rp_beta = lambda: min(1.0, rp_beta_start + self.system.iterations * (1.0 - rp_beta_start) / rp_beta_iterations)

    def __call__(self, priorities):
        '''
        Returns a sample of batches.
        input: priorities - Loss
        output: Weights
        '''
        # calculating importance sampling weights to evade bias
        # these weights are annealed to be more like uniform at the beginning of learning
        weights  = (priorities) ** (-self.rp_beta())
        # these weights are normalized as proposed in the original article to make loss function scale more stable.
        weights /= priorities.min() ** (-self.rp_beta())

        # logs
        self.log("mean weight", np.mean(weights), "weights")
        
        return self.mdp["Weights"](weights)

    def __repr__(self):
        return f"Adds weights to correct bias"