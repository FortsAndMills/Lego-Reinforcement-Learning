from LegoRL.losses.loss import Loss
from LegoRL.core.cache import storage_cached
from LegoRL.core.reference import Reference

class EntropyLoss(Loss):
    """
    Entropy loss for stochastic policies
    
    Args:
        sampler - RLmodule with "sample" method
        policy - RLmodule transformation, returning Policy

    Provides: loss, batch_loss
    """
    def __init__(self, sampler, policy):
        super().__init__(sampler=sampler)        
        self.policy = Reference(policy)

    @storage_cached("loss")
    def batch_loss(self, storage):
        return self.mdp["Loss"](-self.policy(storage).entropy())
        
    def __repr__(self):
        return f"Calculates entropy penalty for <{self.policy.name}> using rollouts from <{self.sampler.name}>"