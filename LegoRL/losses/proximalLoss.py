from LegoRL.losses.loss import Loss
from LegoRL.core.cache import storage_cached
from LegoRL.core.reference import Reference

import torch
from LegoRL.utils.namedTensorsUtils import torch_min

class ProximalLoss(Loss):
    """
    Proximal Policy Loss for actor in policy gradient methods
    Based on: https://arxiv.org/abs/1707.06347
    
    Args:
        sampler - RLmodule with "sample" method
        policy - RLmodule with "log_prob" method
        target - RLmodule with"advantage" method
        ppo_clip - clipping rate of pi_new / pi_old fraction, float

    Provides: loss, batch_loss
    """
    def __init__(self, sampler, policy, target, ppo_clip=0.2):
        super().__init__(sampler=sampler)        
        self.policy = Reference(policy)   
        self.target = Reference(target)
        self.ppo_clip = ppo_clip

    @storage_cached("loss")
    def batch_loss(self, storage):
        '''
        Calculates loss for PPO surrogate function.
        input: Storage
        output: Loss
        '''
        advantages = self.target.advantage(storage).scalar().tensor.detach()

        # old policy action probs
        # TODO: this is madness!
        old_log_prob = self.policy(storage.full_rollout).batch(storage.indices).log_prob(storage.actions.tensor.rename(None))
        
        # importance sampling for making an update of current policy using samples from old policy
        # the gradients to policy will flow through the numerator.
        ratio = torch.exp(self.policy.log_prob(storage) - old_log_prob.detach())

        # PPO clipping! Prevents from "too high updates".
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages
        
        return self.mdp["Loss"](-torch_min(surr1, surr2))

    def hyperparameters(self):
        return {"ppo_clip": self.ppo_clip}
        
    def __repr__(self):
        return f"Calculates PPO gradient estimation for <{self.policy.name}> using samples from <{self.sampler.name}>"