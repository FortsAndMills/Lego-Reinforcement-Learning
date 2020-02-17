from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference

import math
import numpy as np
   
class eGreedy(RLmodule):
    """
    Basic e-Greedy exploration strategy.

    Args:
        greedy_policy - RLmodule with "act" method
        epsilon_start - value of epsilon at the beginning, float, from 0 to 1
        epsilon_final - minimal value of epsilon, float, from 0 to 1
        epsilon_decay - degree of exponential damping of epsilon, int

    Provides: act
    """
    def __init__(self, greedy_policy, epsilon_start=1, epsilon_final=0.01, epsilon_decay=500, frozen=False):
        super().__init__(frozen=frozen)
        
        self.greedy_policy = Reference(greedy_policy)
        
        self.hyperparameters = lambda: {"epsilon_start": epsilon_start, "epsilon_final": epsilon_final, "epsilon_decay": epsilon_decay}
        self.epsilon_by_frame = lambda: epsilon_final + (epsilon_start - epsilon_final) * \
                                        math.exp(-1. * self.system.iterations / epsilon_decay)

    def act(self, storage):
        '''
        For each environment instance selects random action with eps probability
        input: Storage
        '''
        eps = self.epsilon_by_frame()
        self.log("eps", eps, "annealing hyperparameter")

        explore = np.random.uniform(0, 1, size=storage.batch_size) <= eps
        
        self.greedy_policy.act(storage)
        
        if self.frozen:
            self.debug("frozen, does nothing.")
        elif explore.any():
            self.debug("mixing some exploration in...")
            new_actions = storage.actions.numpy
            new_actions[explore] = np.array([self.mdp.action_space.sample() for _ in range(explore.sum())])
            storage.actions.numpy = new_actions 

    def __repr__(self):
        return f"Acts randomly with eps-probability, otherwise calls <{self.greedy_policy.name}>"
