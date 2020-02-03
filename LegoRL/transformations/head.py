from LegoRL.transformations.transformation import Transformation
from LegoRL.core.reference import Reference

import torch

class Head(Transformation):
    """
    Provides common interface for head classes, connected to some other transformation
    
    Args:
        backbone - RLmodule with methods "__call__" and "output_representation"
        network - nn.Module class for head, which accepts (input_shape, output_shape) as
                      constructor parameters
        representation - class, inherited from Representation.
    """
    def __init__(self, backbone, network=torch.nn.Linear, representation=None):
        super().__init__(network=network)

        self.backbone = Reference(backbone)
        self._output_representation = representation

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
        return f"Head, connected to <{self.backbone.name}>, modeling {self._output_representation.__name__}"