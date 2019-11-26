from LegoRL.losses.loss import Loss
from LegoRL.core.cache import batch_cached
from LegoRL.core.composed import Reference

class EntropyLoss(Loss):
    """
    Entropy loss for stochastic policies
    
    Args:
        rollout - RLmodule with "sample" method
        policy - RLmodule with "distribution" method

    Provides: loss, batch_loss
    """
    def __init__(self, rollout, policy, *args, **kwargs):
        super().__init__(sampler=rollout, *args, **kwargs)
        
        self.policy = Reference(policy)

    @batch_cached("loss")
    def batch_loss(self, rollout):
        return -self.policy.distribution(rollout).entropy()
        
    def __repr__(self):
        return f"Calculates entropy penalty for <{self.policy.name}> using rollouts from <{self.sampler.name}>"