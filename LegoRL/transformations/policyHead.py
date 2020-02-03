from LegoRL.transformations.head import Head
from LegoRL.representations.policy import Policy

class PolicyHead(Head):
    """
    Provides a head for policy.
    Provides: act
    """
    def __init__(self, representation=Policy, *args, **kwargs):
        super().__init__(representation=representation, *args, **kwargs)
        
    def act(self, storage):
        self.debug("received act query.", open=True)
        storage.actions = self(storage).sample()        
        self.debug(close=True)

    def log_prob(self, storage):
        '''
        Calculates log probabilities of actions from storage
        input: Storage
        output: FloatTensor, (*batch_shape)
        '''
        #NamedTensors issue
        return self(storage).log_prob(storage.actions.tensor.rename(None))
