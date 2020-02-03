from LegoRL.losses.loss import Loss
from LegoRL.core.cache import storage_cached
from LegoRL.core.reference import Reference

class DynamicsLoss(Loss):
    """
    Loss for dynamics models (aka curiosity)
    
    Args:
        sampler - RLmodule with "sample" method
        model - RLmodule with "curiosity" method

    Provides: loss, batch_loss
    """
    def __init__(self, sampler, model):
        super().__init__(sampler=sampler)        
        self.sampler = Reference(sampler)
        self.model = Reference(model)

    @storage_cached("loss")
    def batch_loss(self, storage):
        return self.model.curiosity(storage)
        
    def __repr__(self):
        return f"Calculates loss for <{self.sampler.name}> using samples from <{self.sampler.name}>"