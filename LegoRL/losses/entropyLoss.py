from LegoRL.losses.loss import Loss

class EntropyLoss(Loss):
    """
    Entropy loss for stochastic policies
    """
    def batch_loss(self, policy, *args, **kwargs):
        '''
        input: Policy
        output: Loss
        '''
        return self.mdp["Loss"](-policy.entropy())
        
    def __repr__(self):
        return f"Calculates entropy penalty"