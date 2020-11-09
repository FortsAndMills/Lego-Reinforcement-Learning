from LegoRL.core.RLmodule import RLmodule
from LegoRL.representations.standard import Reward

import numpy as np
from copy import copy

class IntrinsicMotivation(RLmodule): 
    """
    Adds intrinsic motivation to the rewards from other runner.
    
    Args:
        runner - RLmodule with "sample" method, "done", "episodes_done" properties.
        motivations - list of RLmodules with "curiosity" method.
        coeffs - weight of intrinsic reward, float.
        regime - "add", "replace", "stack", str
        cold_start - iterations before adding, int.

    Provides:
        sample - returns Storage with transitions from interaction.
    """
    def __init__(self, sys, weight=1.0):
        super().__init__(sys)
        self.weight = weight

    def add(self, intrinsic_rewards, is_start):
        '''
        Adds intrinsic motivation to samples from runner.
        input: Reward
        input: Flag
        output: Reward
        '''
        if not hasattr(self, "intrinsic_R"):
            self.intrinsic_R = self.mdp[Reward](np.zeros_like(intrinsic_rewards.numpy))
        else:
            for res in self.intrinsic_R[is_start].numpy:
                self.log(self.name + " curiosity", res, "reward")
                    
        self.intrinsic_R[is_start] = 0
        self.intrinsic_R += self.weight * intrinsic_rewards
        
        return self.weight * intrinsic_rewards

    def hyperparameters(self):
        return {"weight": self.weight}

    def __repr__(self):
        return f"Adds intrinsic rewards"