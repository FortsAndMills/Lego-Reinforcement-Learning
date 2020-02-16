from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference
from LegoRL.representations.standard import State
from LegoRL.transformations.transformation import Transformation

import torch

class Head(Transformation):
    """
    Provides common interface for head classes, connected to some other transformation
    
    Args:
        backbone - RLmodule with methods "__call__" and "output_representation" or None
        network - nn.Module class for head, which accepts (input_shape, output_shape) as
                      constructor parameters
        representation - class, inherited from Representation.
    """
    def __init__(self, backbone=None, network=torch.nn.Linear, representation=None):
        super().__init__(network=network)

        self.backbone = Reference(backbone)
        self._output_representation = representation

    def _initialize(self):
        if self.backbone is None:
            self.backbone = lambda storage, which: storage.crop_states(which)
            self.backbone.output_representation = self.mdp[State]
        super()._initialize()

    @property
    def input_representation(self):
        return self.backbone.output_representation

    @property
    def output_representation(self):
        assert self._output_representation is not None
        return self.mdp[self._output_representation]

    def _get_input(self, storage, which):
        return self.backbone(storage, which).tensor

    def __repr__(self):
        connected = "connected to <{self.backbone.name}>, " if isinstance(self.backbone, RLmodule) else "" 
        return "Head, " + connected + f"modeling {self._output_representation._defaultname()}"