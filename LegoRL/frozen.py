from .backbone import *
from copy import deepcopy

def Frozen(parclass):  
    '''
    Class fabric for target network heuristic implementation.
    
    Args:
        source - RLmodule with "net" property to be copied
    ''' 
    class Frozen(parclass): 
        def __init__(self, source, timer=100, *args, **kwargs):
            super().__init__(timer=timer, backbone=None, *args, **kwargs)
            self.source = Reference(source)
        
        def initialize(self):
            self.net = NetworkWithCache(
                    system = self.system,
                    backbone_name = self.source.net.backbone_name + "__[frozencopy]",
                    head_name = self.name,
                    backbone = deepcopy(self.source.net.backbone),
                    head = deepcopy(self.source.net.head[0]),
                    hat = deepcopy(self.source.net.head[1]))

        def unfreeze(self):
            '''copy policy net weights to target net'''
            self.debug("updates target network.")
            self.net.load_state_dict(self.source.net.state_dict())

        def iteration(self):
            self.unfreeze()

        def load(self, name):  # TODO: what if source was not loaded yet?
            self.unfreeze()

        def __repr__(self):
            return f"Copy of {self.source.name}, updated each {self.timer} iteration"
    return Frozen
