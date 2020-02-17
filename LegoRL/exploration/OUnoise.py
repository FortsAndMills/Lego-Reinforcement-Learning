from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference

import numpy as np
   
class OUNoise(RLmodule):
    """
    Ornstein-Uhlenbeck noise process.
    
    Args:
        greedy_policy - RLmodule with "act" method
        OU_theta - float
        OU_sigma - float

    Provides: act
    """
    def __init__(self, greedy_policy, OU_theta=0.15, OU_sigma=0.2, frozen=False):
        super().__init__(frozen=frozen)
        
        self.greedy_policy = Reference(greedy_policy)
        
        self.OU_theta = OU_theta
        self.OU_sigma = OU_sigma
        self._last_seen_id = None

    def act(self, storage):
        '''
        Adds noise to actions performed by greedy policy.
        input: Storage
        '''
        self.greedy_policy.act(storage)
        
        if self.frozen:
            self.debug("frozen, does nothing")
        else:
            self.debug("adding some noise in actions...")
            
            # zeroing noise when reset
            if storage.id - 1 != self._last_seen_id:
                self.noise = np.zeros((storage.batch_size, *self.mdp.action_shape))
            self._last_seen_id = storage.id

            # zeroing noise when done
            self.noise[storage.just_done.numpy] = np.zeros(self.mdp.action_shape)

            # adding noise
            self.noise -= self.config.OU_theta * self.noise
            self.noise += self.config.OU_sigma * np.random.normal((storage.batch_size, *self.mdp.action_shape))
            storage.actions.numpy += self.noise

    def hyperparameters(self):
        return {"OU_theta": self.OU_theta, "OU_sigma": self.OU_sigma}
        
    def __repr__(self):
        return f"Adds Ornstein–Uhlenbeck noise to actions of <{self.greedy_policy.name}>"
