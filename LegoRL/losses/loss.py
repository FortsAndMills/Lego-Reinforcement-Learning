from LegoRL.core.RLmodule import RLmodule

class Loss(RLmodule):
    """
    Common interface for loss functions
    
    Provides: loss, batch_loss
    """
    def __init__(self, par, weight=1):
        super().__init__(par)
        self.weight = weight

    def batch_loss(self, *args, **kwargs):
        '''
        Computes loss for batch
        output: Loss
        '''
        raise NotImplementedError()
    
    def __call__(self, *args, weights=None, **kwargs):
        '''
        Calculates loss for batch from sampler
        output: Tensor, scalar.
        '''
        self.last_batch_loss = self.batch_loss(*args, **kwargs)
        if weights is None:
            loss = self.last_batch_loss.tensor.mean()
        else:
            loss = (self.last_batch_loss * weights).tensor.mean()

        loss = self.weight * loss
        self.log(self.name, loss.detach().cpu().numpy(), f"{self.name} loss")
        return loss

    def hyperparameters(self):
        return {"weight": self.weight}