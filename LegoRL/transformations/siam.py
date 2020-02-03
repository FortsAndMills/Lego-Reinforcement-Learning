from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference
from LegoRL.buffers.storage import stack

import torch

class Siam(RLmodule):
    '''
    Siam backbone taking state and next state as input,
    transforming them using the same backbone and concatentating output.
    
    Args:
        backbone - RLmodule, providing "__call__" and "output_representation" methods.

    Provides: __call__, output_representation
    '''

    def __init__(self, backbone):
        super().__init__()        
        self.backbone = Reference(backbone)

    @property
    def output_representation(self):
        '''output: Representation class'''
        return self.mdp[Embedding(self.backbone.output_representation.embedding_size * 2)]

    def __call__(self, storage):
        output = self.backbone(storage, Which.all)
        return Embedding.cat([output.crop(Which.current), output.crop(Which.next)])

    def __repr__(self):
        return f"Siam backbone constructed from <{self.backbone.name}>"