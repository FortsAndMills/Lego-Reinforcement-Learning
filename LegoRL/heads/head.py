from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.composed import Reference
from LegoRL.core.cache import cached
from LegoRL.representations.representation import Representation

import torch

class Head(RLmodule, torch.nn.Module):
    """
    Provides common interface for head classes
    
    Args:
        backbone - RLmodule for backbone with "mount_head"
        network - nn.Module class for head, which accepts (input_shape, output_shape) as
                      constructor parameters
        representation - class, inherited from Representation.
    """
    def __init__(self, backbone, network=torch.nn.Linear, representation=Representation, *args, **kwargs):
        RLmodule.__init__(self, *args, **kwargs)
        torch.nn.Module.__init__(self)

        self.backbone = Reference(backbone)
        self.net = network
        self.representation = representation

    def _initialize(self):
        print(f"Adding new head <{self.name}> to <{self.backbone.name}>:")
        print(f"  Input shape is {self.input_shape}")
        
        feature_size = self.backbone.feature_size(self.input_shape)
        print(f"  Backbone feature size is {feature_size}")

        output_size = self.representation.shape(self.system)
        print(f"  Desired output is {output_size}")   

        self.net = self.net(feature_size, output_size.numel()).to(self.system.device)
        self.backbone.mount_head(self.net)

    @property
    def input_shape(self):
        return self.system.observation_shape

    @cached
    def forward(self, *input, **kwargs):
        features = self.backbone(*input, **kwargs)
        output = self.net(features)
        return self.representation.from_linear(self.system, output)

    def __repr__(self):
        return f"Head, connected to <{self.backbone.name}>, modeling {self.representation.__name__}"