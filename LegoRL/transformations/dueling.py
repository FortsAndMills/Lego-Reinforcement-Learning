from LegoRL.core.reference import Reference
from LegoRL.representations.policy import Policy
from LegoRL.buffers.storage import Which
from LegoRL.transformations.qualityHead import QualityHead

import torch

class Dueling(QualityHead):
    """
    Dueling DQN head. Represents Q-function using two heads, Advantage and Value,
    which are aggregated using heuristic
        Q = V + A - A.mean(dim="actions")
    Based on: https://arxiv.org/abs/1511.06581

    Args:
        value_head - RLmodule, providing "V"

    Provides: act, V, Q, estimate
    """
    def __init__(self, value_head, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.value_head = Reference(value_head)

    def Q(self, storage, which=Which.current):
        '''
        Calculates Q function
        input: Storage
        input: Which
        output: Q
        '''
        q = super().Q(storage, which)
        
        # heuristic transform to Advantage
        uniform_policy = self.mdp[Policy].uniform()
        a = q.subtract_v(q.value(uniform_policy))
        
        # adding V
        v = self.value_head.V(storage, which)
        return a.add_v(v)

    def __repr__(self):
        return super().__repr__() + f' in dueling form (V + A - A.mean()) where V comes from <{self.value_head.name}>'