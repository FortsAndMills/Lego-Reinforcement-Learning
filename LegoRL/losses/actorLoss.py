from LegoRL.losses.loss import Loss
from LegoRL.core.cache import batch_cached
from LegoRL.core.composed import Reference

class ActorLoss(Loss):
    """
    ActorCritic loss for actor in policy gradient methods
    Based on: https://arxiv.org/abs/1602.01783
    
    Args:
        rollout - RLmodule with "sample" method
        policy - RLmodule with "distribution" method
        target - RLmodule with "returns" method
        baseline - RLmodule with "V" method

    Provides: loss, batch_loss
    """
    def __init__(self, rollout, policy, target, baseline, *args, **kwargs):
        super().__init__(sampler=rollout, *args, **kwargs)
        
        self.policy = Reference(policy)
        self.target = Reference(target)
        self.baseline = Reference(baseline)

    @batch_cached("loss")
    def batch_loss(self, rollout):
        '''
        Calculates loss for surrogate function with same gradient
        as in Policy Gradient Theorem.
        input: rollout - Rollout
        output: Tensor, (*batch_shape)
        '''
        advantages = self.target.returns(rollout).subtract_v(self.baseline.V(rollout))
        log_probs = self.policy.distribution(rollout).log_prob(rollout.actions.rename(None))
        return -log_probs * advantages.scalar().detach()
        
    def __repr__(self):
        return f"Calculates gradient estimation for <{self.policy.name}> using returns from <{self.target.name}>, baseline from <{self.baseline.name}> and rollouts from <{self.sampler.name}>"