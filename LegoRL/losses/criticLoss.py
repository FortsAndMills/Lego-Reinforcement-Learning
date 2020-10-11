from LegoRL.losses.loss import Loss

import torch

class CriticLoss(Loss):
    """
    Critic loss based on comparing output with target
    for solving Bellman equations.
    """
    def batch_loss(self, prediction, target, *args, **kwargs):
        '''
        input: prediction - V
        input: target - V
        output: Loss
        '''
        return prediction.compare(target.detach())
        
    def __repr__(self):
        return f"Calculates TD loss"