from LegoRL.transformations.criticHead import CriticHead
from LegoRL.representations.Q import Q
from LegoRL.buffers.storage import Which

import torch

class QualityHead(CriticHead):
    """
    Provides a head for Q-function.

    Provides: act, V, Q, estimate
    """        
    def __init__(self, representation=Q, *args, **kwargs):
        super().__init__(representation=representation, *args, **kwargs)

    def act(self, storage):
        self.debug("received act query.", open=True)        
        with torch.no_grad():
            storage.actions = self.Q(storage).greedy()            
        self.debug(close=True)

    def V(self, storage, which=Which.current, policy=None):
        '''
        Calculates V function from Q function for given policy or V = maxQ if policy is None
        input: Storage
        input: Which
        input: Policy or None
        output: V
        '''
        return self.Q(storage, which).value(policy)

    def Q(self, storage, which=Which.current):
        return self(storage, which)

    def estimate(self, storage):
        '''
        Estimates batch state-action pairs as Q(s, a)
        input: Storage
        output: V
        '''
        return self.Q(storage).gather(storage.actions)