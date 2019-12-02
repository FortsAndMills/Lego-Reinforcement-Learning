from LegoRL.core.composed import Reference
from LegoRL.losses.loss import Loss
from LegoRL.core.cache import batch_cached

import torch

class CriticLoss(Loss):
    """
    Q-learning loss based on policy iteration
    for solving Bellman equations.
    
    Args:
        sampler - RLmodule with "sample" method
        critic - RLmodule with "estimate" method
        target - RLmodule with "returns" method

    Provides: loss, batch_loss
    """
    def __init__(self, sampler, critic, target, *args, **kwargs):
        super().__init__(sampler=sampler, *args, **kwargs)
        
        self.critic = Reference(critic)
        self.target = Reference(target) 
    
    @batch_cached("loss")
    def batch_loss(self, batch):
        '''
        Calculates loss for batch based on TD-error from DQN algorithm.
        input: batch - Batch
        output: Tensor, (*batch_shape)
        '''
        q = self.critic.estimate(batch)
        with torch.no_grad():
            target = self.target.returns(batch)

        return q.compare(target)
        
    def __repr__(self):
        return f"Calculates TD loss for <{self.critic.name}> using <{self.target.name}> as target calculator and data from <{self.sampler.name}>"