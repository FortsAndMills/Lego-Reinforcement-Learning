from LegoRL.heads.head import Head
from LegoRL.representations.policy import Policy

import torch

class PolicyHead(Head):
    """
    Provides a head for policy.
    Provides: act, distribution
    """
    def __init__(self, representation=Policy, *args, **kwargs):
        super().__init__(representation=representation, *args, **kwargs)
        
    def act(self, transitions):
        self.debug("received act query.", open=True)        
        transitions.to_torch(self.system)

        transitions.actions = self.distribution(transitions).sample()

        transitions.to_numpy()
        self.debug(close=True)

    def distribution(self, batch):
        return self(batch.states, storage=batch, cache_name="state")