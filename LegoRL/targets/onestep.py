from LegoRL.core.RLmodule import RLmodule

import torch

class OneStep(RLmodule):
    """
    One step value estimation
    V = r + gamma * V
    """
    def __call__(self, evaluator, next_states, rewards, discounts, *args, **kwargs):
        '''
        input: evaluator - callable, returning V
        input: State
        input: Reward
        input: Discount
        output: V
        '''
        with torch.no_grad():
            next_V = evaluator.V(next_states)
            return next_V.one_step(rewards, discounts)

    def __repr__(self):
        return f"Returns one-step approximation"
