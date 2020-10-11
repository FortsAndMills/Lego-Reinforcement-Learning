from LegoRL.representations.discretePolicy import DiscretePolicy
from LegoRL.representations.V import V
from LegoRL.representations.Q import Q
from LegoRL.core.RLmodule import RLmodule

class Dueling(RLmodule):
    """
    Dueling DQN head. Represents Q-function using two heads, Advantage and Value,
    which are aggregated using heuristic
        Q = V + A - A.mean(dim="actions")
    Based on: https://arxiv.org/abs/1511.06581
    """
    def __call__(self, q, v):
        '''
        Calculates Q function
        input: Q
        input: V
        output: Q
        '''        
        # heuristic transform to Advantage
        uniform_policy = self.mdp[DiscretePolicy].uniform()
        a = q.subtract_v(q.value(uniform_policy))
        
        # adding V
        return a.add_v(v)

    def __repr__(self):
        return f'Combines A and V in dueling form (V + A - A.mean())'