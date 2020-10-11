from LegoRL.buffers.storage import Storage
from LegoRL.models.model import Model
from LegoRL.representations.Q import Q

import torch.nn as nn

class QCritic():
    """
    Provides: Q
    """
    def act(self, states, *args, **kwargs): 
        """
        input: State
        output: Storage
        """
        Q = self(states)
        actions = Q.greedy()
        return Storage(actions=actions, Q=Q)

    def V(self, states, policy=None):
        '''
        Calculates V function from Q function for given policy or V = maxQ if policy is None
        input: State
        input: Policy or None (greedy action will be taken)
        output: V
        '''
        return self(states).value(policy)

    def Q(self, states, actions):
        """
        input: State
        input: Action
        output: Q
        """    
        return self(states).gather(actions)

class QNetwork(Model, QCritic):
    """
    Provides a model of Q-function.

    Provides: act, V, Q, estimate
    """        
    def __init__(self, *args, output=Q, **kwargs):
        assert issubclass(output, Q), "Error: output representation must be Q"
        Model.__init__(self, output=output, *args, **kwargs)
        QCritic.__init__(self)    