from LegoRL.losses.loss import Loss

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
    def __init__(self, sys, ppo_clip=0.2, *args, **kwargs):
        super().__init__(sys, *args, **kwargs)
        self.ppo_clip = ppo_clip

    def batch_loss(self, policy, old_policy, actions, advantages, *args, **kwargs):
        '''
        Calculates loss for PPO surrogate
        input: Policy
        input: old_policy - Policy
        input: Action
        input: advantages - V
        output: Loss
        '''
        # importance sampling for making an update of current policy using samples from old policy
        # the gradients to policy will flow through the numerator.
        ratio = torch.exp(policy.log_prob(actions) - old_policy.log_prob(actions).detach())

        # detach advantages
        advantages = advantages.tensor.detach()
        assert ratio.shape == advantages.shape

        # PPO clipping! Prevents from "too high updates".
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip) * advantages
        
        return self.mdp["Loss"](-torch_min(surr1, surr2))

    def hyperparameters(self):
        return {"ppo_clip": self.ppo_clip}
        
    def __repr__(self):
        return f"Calculates PPO gradient estimation"