from LegoRL.losses.loss import Loss
from LegoRL.core.cache import storage_cached
from LegoRL.core.reference import Reference

class ActorLoss(Loss):
    """
    ActorCritic loss for actor in policy gradient methods
    Based on: https://arxiv.org/abs/1602.01783
    
    Args:
        sampler - RLmodule with "sample" method
        policy - RLmodule with "log_prob" method
        target - RLmodule with "advantage" method

    Provides: loss, batch_loss
    """
    def __init__(self, sampler, policy, target):
        super().__init__(sampler=sampler)        
        self.policy = Reference(policy)
        self.target = Reference(target)

    @storage_cached("loss")
    def batch_loss(self, storage):
        '''
        Calculates loss for surrogate function with same gradient as in Policy Gradient Theorem.
        input: Storage
        output: Loss
        '''
        advantages = self.target.advantage(storage).scalar().tensor.detach()
        log_probs = self.policy.log_prob(storage)
        return self.mdp["Loss"](-log_probs * advantages)
        
    def __repr__(self):
        return f"Calculates gradient estimation for <{self.policy.name}> using advantages from <{self.target.name}> and rollouts from <{self.sampler.name}>"