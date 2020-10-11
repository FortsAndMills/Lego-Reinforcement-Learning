from LegoRL.models.model import Model
from LegoRL.representations.V import V
from LegoRL.representations.standard import State, Action

class ForwardQNetwork(Model):
    """
    Provides a model of Q-function, which takes action as input

    Provides: Q
    """        
    def __init__(self, *args, input=[State, Action], output=V, **kwargs):
        assert issubclass(output, V), "Error: output representation must be V"
        super().__init__(input=input, output=output, *args, **kwargs)

    def Q(self, states, actions):
        """
        input: State
        input: Action
        output: V
        """    
        return self(states, actions)