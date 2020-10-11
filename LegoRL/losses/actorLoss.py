from LegoRL.losses.loss import Loss

class ActorLoss(Loss):
    """
    ActorCritic loss for actor in policy gradient methods
    Based on: https://arxiv.org/abs/1602.01783

    """
    def batch_loss(self, policy, actions, advantages, *args, **kwargs):
        '''
        Calculates loss for surrogate function with same gradient as in Policy Gradient Theorem.
        input: Policy
        input: Action
        input: V
        output: Loss
        '''
        advantages = advantages.scalar().tensor.detach()
        log_probs = policy.log_prob(actions)
        return self.mdp["Loss"](-log_probs * advantages)
        
    def __repr__(self):
        return f"Calculates policy loss"