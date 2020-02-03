from LegoRL.transformations.head import Head
from LegoRL.buffers.storage import Which
from LegoRL.representations.standard import Action

class DeterministicPolicyHead(Head):
    """
    Provides a head for deterministic policy.
    Provides: act
    """
    def __init__(self, *args, **kwargs):
        super().__init__(representation=Action, *args, **kwargs)

    def act(self, storage):
        self.debug("received act query.", open=True)
        storage.actions = self(storage, Which.current)        
        self.debug(close=True)
