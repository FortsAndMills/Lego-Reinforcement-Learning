from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.composed import Reference

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
    def __init__(self, greedy_policy, epsilon_start=1, epsilon_final=0.01, epsilon_decay=500, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.greedy_policy = Reference(greedy_policy)
        self.epsilon_by_frame = lambda: epsilon_final + (epsilon_start - epsilon_final) * \
                                        math.exp(-1. * self.system.iterations / epsilon_decay)

    def act(self, transitions):
        '''
        For each environment instance selects random action with eps probability
        Input: transitions - TransitionBatch
        Output: action - list, (batch_size)
        '''
        eps = self.epsilon_by_frame()
        self.log("eps", eps, "training iteration", "annealing hyperparameter")

        explore = np.random.uniform(0, 1, size=transitions.batch_size) <= eps
        
        self.greedy_policy.act(transitions)
        
        if explore.any():
            self.debug("mixing some exploration in...")
            transitions.actions[explore] = np.array([self.system.action_space.sample() for _ in range(explore.sum())])
        
    def __repr__(self):
        return f"Acts randomly with eps-probability, otherwise calls <{self.greedy_policy.name}>"
