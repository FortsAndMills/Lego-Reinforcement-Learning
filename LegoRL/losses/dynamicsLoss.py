from LegoRL.losses.loss import Loss

class DynamicsLoss(Loss):
    """
    Loss for dynamics models
    """
    def batch_loss(self, prediction, target, *args, **kwargs):
        '''
        input: prediction - State, Action or Embedding
        input: target - State, Action or Embedding
        output: Loss
        '''
        return prediction.compare(target.detach())
        
    def __repr__(self):
        return f"Calculates loss"