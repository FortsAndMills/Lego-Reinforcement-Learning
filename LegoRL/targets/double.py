from LegoRL.core.RLmodule import RLmodule

import torch

class Double(RLmodule):
    """
    Value estimation based on decoupled action selection and evaluation:
    V = Q1(argmax Q2)
    """
    def __call__(self, evaluator, selector, next_states, rewards, discounts, *args, **kwargs):
        '''
        input: evaluator - callable, returning Q
        input: selector - callable, returning Q
        input: next_states - State
        input: Reward
        input: Discount
        output: V
        '''
        with torch.no_grad():
            chosen_actions = selector(next_states).greedy()
            next_V = evaluator(next_states).gather(chosen_actions)
            return next_V.one_step(rewards, discounts)

    def __repr__(self):
        return f"Decouples action selection and action evaluation"
