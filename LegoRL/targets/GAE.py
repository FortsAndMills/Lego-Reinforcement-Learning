from LegoRL.core.RLmodule import RLmodule
from LegoRL.buffers.storage import stack

import torch

class GAE(RLmodule):
    """
    Generalized Advantage Estimation (GAE) upgrade of A2C.
    Based on: https://arxiv.org/abs/1506.02438
    
    Args:
        tau - float, from 0 to 1

    Provides: returns, advantage
    """
    def __init__(self, sys, tau=0.95):
        super().__init__(sys)
        self.tau = tau

    def __call__(self, rewards, values, discounts, last_V):
        '''
        Calculates GAE return.
        input: Reward
        input: V
        input: Discount
        input: V
        output: V
        '''
        with torch.no_grad():
            returns = []
            gae = 0
            next_values = last_V
            
            for step in reversed(range(rewards.rollout_length)):
                advantage = next_values.one_step(rewards[step], discounts[step]).subtract_v(values[step])
                next_values = values[step]
                
                gae = advantage + gae * discounts[step] * self.tau
                returns.append(gae)
        return stack(returns[::-1])
    
    def hyperparameters(self):
        return {"GAE tau": self.tau}

    def __repr__(self):
        return f"Estimates GAE advantages"