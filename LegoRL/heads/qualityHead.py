from LegoRL.heads.criticHead import CriticHead
from LegoRL.representations.Q import Q
from LegoRL.representations.V import V

import torch

class QualityHead(CriticHead):
    """
    Provides a head for Q-function.

    Provides: act, V, Q, estimate
    """        
    def __init__(self, representation=Q(V), *args, **kwargs):
        super().__init__(representation=representation, *args, **kwargs)

    def act(self, transitions):
        self.debug("received act query.")

        transitions.to_torch(self.system)
        
        with torch.no_grad():
            transitions.actions = self.Q(transitions).greedy()

        transitions.to_numpy()

    def V(self, batch, of="state", policy=None):
        '''
        Calculates V function from Q function for given policy or V = maxQ if policy is None
        input: Batch
        input: of - str, "state", "next state" or "last state"
        input: policy - FloatTensor, (*batch_shape, num_actions)
        output: V
        '''
        return self.Q(batch, of).value(policy)

    def Q(self, batch, of="state"):
        '''
        Calculates Q function
        input: Batch
        input: of - str, "state", "next state" or "last state"
        output: Q
        '''
        return self._evaluate(batch, of)

    def estimate(self, batch):
        '''
        Estimates batch state-action pairs as Q(s, a)
        input: Batch
        output: V
        '''
        return self.Q(batch).gather(batch.actions)

    #TODO: remove this?
    def advantage(self, batch):
        '''
        Estimates A(s, a)
        input: Batch
        output: V
        '''
        q = self.Q(batch)
        return q.gather(batch.actions).subtract_v(q.value())