from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.composed import Reference
from LegoRL.core.cache import batch_cached

class Loss(RLmodule):
    """
    Common interface for loss functions

    Args:
        sampler - RLmodule with "sample" property
    
    Provides: loss, batch_loss
    """
    def __init__(self, sampler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = Reference(sampler)

    @batch_cached("loss")
    def batch_loss(self, batch):
        '''
        Computes loss for batch
        input: batch - Batch
        output: Tensor, (*batch_shape)
        '''
        raise NotImplementedError()
    
    def loss(self):
        '''
        Calculates loss for batch from sampler
        output: Tensor, scalar.
        '''
        batch = self.sampler.sample()
        if batch is None:
            self.debug("no batch is found, loss is None.")
            return

        self.debug("starts loss calculation.", open=True)
        self.batch_loss(batch)
        self.debug("finished loss calculation.", close=True)

        # averaging loss (there might be weights in the batch)      
        return batch.average(self.name + " loss")