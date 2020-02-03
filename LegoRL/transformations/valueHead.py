from LegoRL.transformations.criticHead import CriticHead
from LegoRL.representations.V import V
from LegoRL.buffers.storage import Which

class ValueHead(CriticHead):
    """
    Provides a head for V-function.
    Provides: V, estimate
    """
    def __init__(self, representation=V, *args, **kwargs):
        super().__init__(representation=representation, *args, **kwargs)

    def V(self, storage, which=Which.current):
        return self(storage, which)

    def estimate(self, storage):
        '''
        Estimates batch state-action pairs (WTF) as V(s)
        input: Storage
        output: V
        '''
        return self.V(storage)