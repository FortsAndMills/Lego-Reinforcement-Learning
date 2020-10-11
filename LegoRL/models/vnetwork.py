from LegoRL.models.model import Model
from LegoRL.representations.V import V
from LegoRL.representations.Q import Q

class VCritic():
    """
    Provides: V
    """
    def V(self, states):
        '''
        input: State
        output: V
        '''
        return self(states)

class VNetwork(Model, VCritic):
    """
    Provides a head for V-function.
    Provides: V, estimate
    """
    def __init__(self, *args, output=V, **kwargs):
        assert not issubclass(output, Q), "Error: output representation can't be Q; use QNetwork for this"
        Model.__init__(self, output=output, *args, **kwargs)
        VCritic.__init__(self)