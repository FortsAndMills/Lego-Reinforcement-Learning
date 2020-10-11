from LegoRL.losses.loss import Loss

class DeterministicActorLoss(Loss):
    """
    TODO
    """
    def batch_loss(self, V, *args, **kwargs):
        '''
        Loss for direct policy improvement through critic.
        input: V
        output: Loss
        '''
        return self.mdp["Loss"](-V.scalar().tensor)
        
    def __repr__(self):
        return f"Calculates deterministic policy loss"