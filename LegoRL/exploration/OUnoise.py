from LegoRL.core.RLmodule import RLmodule

import numpy as np
   
class OUnoise(RLmodule):
    """
    Ornstein-Uhlenbeck noise process.
    
    Args:
        OU_theta - float
        OU_sigma - float

    Provides: mix
    """
    def __init__(self, sys, OU_theta=0.15, OU_sigma=0.2):
        super().__init__(sys)
        
        self.OU_theta = OU_theta
        self.OU_sigma = OU_sigma

    def __call__(self, actions, is_start):
        '''
        Adds noise to actions performed by greedy policy.
        input: actions - Action
        input: is_start - Flag
        output: Action
        '''        
        # zeroing noise when reset
        if not hasattr(self, "noise"):
            self.noise = np.zeros((actions.batch_size, *self.mdp.action_shape))

        # zeroing noise when done
        self.noise[is_start.numpy] = np.zeros(self.mdp.action_shape)

        # adding noise
        self.noise -= self.OU_theta * self.noise
        self.noise += self.OU_sigma * np.random.normal(size=(actions.batch_size, *self.mdp.action_shape))
        return actions + self.noise

    def hyperparameters(self):
        return {"OU_theta": self.OU_theta, "OU_sigma": self.OU_sigma}
        
    def __repr__(self):
        return f"Adds Ornstein–Uhlenbeck noise to actions"
