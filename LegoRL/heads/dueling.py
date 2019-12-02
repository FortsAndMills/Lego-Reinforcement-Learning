from LegoRL.heads.qualityHead import QualityHead
from LegoRL.core.composed import Reference

import torch

class Dueling(QualityHead):
    """
    Dueling DQN head. Represents Q-function using two heads, Advantage and Value,
    which are aggregated using heuristic
        Q = V + A - A.mean(dim="actions")
    Based on: https://arxiv.org/abs/1511.06581

    Args:
        value_head - RLmodule, providing "v"

    Provides: act, V, Q, estimate
    """
    def __init__(self, value_head, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.value_head = Reference(value_head)

    def Q(self, batch, of="state"):
        '''
        Calculates Q function
        input: Batch
        input: of - str, "state", "next state" or "last state"
        output: Q
        '''
        q = super().Q(batch, of)
        v = self.value_head.V(batch, of)
        
        uniform_policy = torch.ones(self.system.num_actions, names=("actions",)).to(self.system.device) / self.system.num_actions
        a = q.subtract_v(q.value(uniform_policy))
        return a.add_v(v)

    def __repr__(self):
        return super().__repr__() + f' in dueling form (V + A - A.mean()) where V comes from <{self.value_head.name}>'