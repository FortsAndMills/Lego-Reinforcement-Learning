from LegoRL.losses.loss import Loss
from LegoRL.core.cache import batch_cached
from LegoRL.core.composed import Reference

import torch

class OnlineCriticLoss(Loss):
    """
    Online Q-learning for some policy.
    
    Args:
        rollout - RLmodule with "sample" method
        critic - RLmodule with "V" method

    Provides: loss, batch_loss, advantage
    """
    def __init__(self, rollout, critic, *args, **kwargs):
        super().__init__(sampler=rollout, *args, **kwargs)
        self.critic = Reference(critic)

    # TODO: this must be in a separate module.
    def returns(self, rollout):
        '''
        Calculates max trace returns, estimating the V of last state using critic.
        input: rollout - Rollout
        output: V
        '''
        self.debug("starts computing max trace returns")
        with torch.no_grad():
            returns = [self.critic.V(rollout, of="last state")]
            for step in reversed(range(rollout.rollout_length)):
                returns.append(returns[-1].one_step(rollout.at(step)))
            return self.critic.representation.stack(returns[-1:0:-1])

    @batch_cached("advantage")
    def advantage(self, rollout):
        '''
        Calculates advantage using max trace returns.
        input: rollout - Rollout
        output: V
        '''
        self.debug("starts computing max trace advantage")
        return self.returns(rollout).subtract_v(self.critic.V(rollout))

    @batch_cached("loss")
    def batch_loss(self, rollout):
        '''
        Calculates loss for batch based on TD-error from DQN algorithm.
        input: rollout - Rollout
        output: Tensor, (*batch_shape)
        '''
        v = self.critic.V(rollout)
        with torch.no_grad():
            target = self.returns(rollout)

        return v.compare(target)
        
    def __repr__(self):
        return f"Calculates online TD loss for <{self.critic.name}> using data from <{self.sampler.name}>"