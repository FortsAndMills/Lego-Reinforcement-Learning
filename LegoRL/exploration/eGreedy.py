from LegoRL.core.RLmodule import RLmodule
from LegoRL.representations.standard import Action

import math
import numpy as np
   
class eGreedy(RLmodule):
    """
    Basic e-Greedy exploration strategy.

    Args:
        epsilon_start - value of epsilon at the beginning, float, from 0 to 1
        epsilon_final - minimal value of epsilon, float, from 0 to 1
        epsilon_decay - degree of exponential damping of epsilon, int

    Provides: mix
    """
    def __init__(self, sys, epsilon_start=1, epsilon_final=0.01, epsilon_decay=500):
        super().__init__(sys)
        
        self.hyperparameters = lambda: {"epsilon_start": epsilon_start, "epsilon_final": epsilon_final, "epsilon_decay": epsilon_decay}
        self.get_epsilon = lambda: epsilon_final + (epsilon_start - epsilon_final) * \
                                        math.exp(-1. * self.system.iterations / epsilon_decay)

    def __call__(self, actions : Action, *args, **kwargs):
        '''
        Substitues actions to random actions with eps probability
        '''
        #assert isinstance(actions, Action), "Error: egreedy got not actions as input!"

        eps = self.get_epsilon()
        self.log("eps", eps, "annealing hyperparameter")

        explore = np.random.uniform(0, 1, size=actions.batch_size) <= eps
        
        new_actions = actions.numpy.copy()       
        if explore.any():  
            new_actions[explore] = np.array([self.mdp.action_space.sample() for _ in range(explore.sum())])
        
        return self.mdp[Action](new_actions)

    def __repr__(self):
        return f"Mixes actions with random actions"
