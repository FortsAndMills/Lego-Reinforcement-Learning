from LegoRL.core.composed import Reference
from LegoRL.core.cache import cached
from LegoRL.core.backbone import Backbone

from copy import deepcopy

def Frozen(parclass):  
    '''
    Class fabric for target network heuristic implementation.
    
    Args:
        source - RLmodule with "net" property.
    '''
    class Frozen(parclass):
        def __init__(self, source, timer=100, *args, **kwargs):
            super().__init__(timer=timer, network=None, *args, **kwargs)
            self.source = Reference(source)

        def _initialize(self):  # TODO: what if source was not initialized yet?
            self.net = deepcopy(self.source.net)
            if hasattr(self.source, "representation"):
                self.representation = self.source.representation

        def unfreeze(self):
            '''Copies weights from source'''
            self.debug("updates frozen network.")
            self.net.load_state_dict(self.source.net.state_dict())

        def iteration(self):
            self.unfreeze()

        def _save(self, name):
            pass

        def _load(self, name):  # TODO: what if source was not loaded yet?
            self.unfreeze()

        @classmethod
        def defaultname(cls):
            '''output: default name of module, str'''
            return cls.__name__ + parclass.__name__

        def __repr__(self):
            return f"Copy of <{self.source.name}>, updated each {self.timer} iteration"
    return Frozen
