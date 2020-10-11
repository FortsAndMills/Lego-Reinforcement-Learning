from LegoRL.core.RLmodule import RLmodule

import torch
from LegoRL.utils.namedTensorsUtils import torch_min

class Twin(RLmodule):
    """
    TODO
    """
    def __call__(self, V1, V2):
        '''
        input: V1
        input: V2
        output: V
        '''
        return type(V1)(torch_min(V1.tensor, V2.tensor))

    def __repr__(self):
        return f"Ensembles two value functions by taking minimum"
