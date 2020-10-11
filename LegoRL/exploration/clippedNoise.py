from LegoRL.core.RLmodule import RLmodule

import numpy as np
   
class ClippedNoise(RLmodule):
    """
    Adds clipped noise
    
    Args:
        sigma - float
        cliprange - float

    Provides: mix
    """
    def __init__(self, par, sigma=0.2, cliprange=0.2):
        super().__init__(par)
        
        self.cliprange = cliprange
        self.sigma = sigma

    def __call__(self, actions):
        '''
        Adds clipped noise to actions
        input: Action
        output: Action
        '''
        # adding noise
        noise = np.random.normal(size=(actions.batch_size, *self.mdp.action_shape))
        noise = np.clip(self.sigma * noise, -self.cliprange, self.cliprange)
        return actions + noise

    def hyperparameters(self):
        return {"cliprange": self.cliprange, "sigma": self.sigma}
        
    def __repr__(self):
        return f"Adds clipped noise to actions"
