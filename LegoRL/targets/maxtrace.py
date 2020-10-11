from LegoRL.core.RLmodule import RLmodule
from LegoRL.buffers.storage import stack

import torch

class MaxTrace(RLmodule):
    """
    MaxTrace return estimator.
    Q(s, a) ~ r(s') + r(s'') + ... V(s_{last})
    """
    def __call__(self, rewards, discounts, last_V):
        '''
        Calculates max trace returns, estimating the V of last state using critic.
        input: Reward
        input: Discount
        input: V
        output: V
        '''
        returns = [last_V]
        for step in reversed(range(rewards.rollout_length)):
            returns.append(returns[-1].one_step(rewards[step], discounts[step]))
        
        return stack(returns[-1:0:-1])
        
    def __repr__(self):
        return f"Estimates maxtrace returns"