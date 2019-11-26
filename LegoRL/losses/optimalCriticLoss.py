from LegoRL.core.composed import Reference
from LegoRL.losses.loss import Loss
from LegoRL.core.cache import batch_cached

import torch

class OptimalCriticLoss(Loss):
    """
    Classic deep Q-learning algorithm (DQN).
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        sampler - RLmodule with "sample" method
        critic - RLmodule with "estimate" method
        target_critic - RLmodule with "V" method

    Provides: loss, batch_loss
    """
    def __init__(self, sampler, critic, target_critic=None, *args, **kwargs):
        super().__init__(sampler=sampler, *args, **kwargs)
        
        self.critic = Reference(critic)
        self.target_critic = Reference(target_critic or critic)

    def estimate_next_state(self, batch):
        '''
        Calculates estimation of V* for next state.
        input: Batch
        output: FloatTensor, (*batch_shape, *value_shape)
        '''
        return self.target_critic.V(batch, of="next state")      
    
    @batch_cached("loss")
    def batch_loss(self, batch):
        '''
        Calculates loss for batch based on TD-error from DQN algorithm.
        input: batch - Batch
        output: Tensor, (*batch_shape)
        '''
        q = self.critic.estimate(batch)
        with torch.no_grad():
            target = self.estimate_next_state(batch).one_step(batch)

        return q.compare(target)
        
    def __repr__(self):
        return f"Calculates one-step optimal TD loss for <{self.critic.name}> using <{self.target_critic.name}> as target calculator and data from <{self.sampler.name}>"