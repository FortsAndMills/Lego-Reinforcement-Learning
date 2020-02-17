from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference
from LegoRL.transformations.transformation import Transformation

from copy import deepcopy

def Frozen(parclass):  
    '''
    Class fabric for target network heuristic implementation.
    
    Args:
        source - RLmodule with "net" property.
    '''
    assert issubclass(parclass, Transformation), "Can freeze only RLmodules inherited from Transformation"

    class Frozen(parclass):
        def __init__(self, source, timer=100, frozen=False, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.timer = timer
            self.frozen = frozen
            self.source = Reference(source)

        def _initialize(self):
            super()._initialize()
            self.source.initialize()
            self.net = deepcopy(self.source.net)
            self._output_representation = self.source._output_representation

        def unfreeze(self):
            '''Copies weights from source'''
            self.debug("updates frozen network.")
            self.net.load_state_dict(self.source.net.state_dict())

        def _iteration(self):
            self.unfreeze()

        def _save(self, folder_name):
            pass

        def _load(self, name):  # TODO: what if source was not loaded yet?
            self.unfreeze()

        @classmethod
        def _default_name(cls):
            return cls.__name__ + parclass.__name__

        def hyperparameters(self):
            return {"timer": self.timer}

        def __repr__(self):
            return f"Copy of <{self.source.name}>, updated each {self.timer} iteration"
    return Frozen
