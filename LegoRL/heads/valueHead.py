from LegoRL.heads.criticHead import CriticHead
from LegoRL.representations.V import V

class ValueHead(CriticHead):
    """
    Provides a head for V-function.
    Provides: V, estimate
    """
    def __init__(self, representation=V, *args, **kwargs):
        assert issubclass(representation, V), "Representation must be inherited from V"
        super().__init__(representation=representation, *args, **kwargs)

    def V(self, batch, of="state"):
        '''
        Calculates V function.
        input: Batch
        input: of - str, "state", "next state" or "last state"
        output: V
        '''
        return self._evaluate(batch, of)

    def estimate(self, batch):
        '''
        Estimates batch state-action pairs as r + V(s')
        input: Batch
        output: V
        '''
        return self.V(batch, of="next state").one_step(batch)