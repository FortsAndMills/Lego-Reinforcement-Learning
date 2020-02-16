from LegoRL.core.reference import Reference
from LegoRL.transformations.backbone import Backbone
from LegoRL.representations.standard import Embedding

import torch
import numpy as np

# TODO: think
class Distillated(Backbone):
    """
    State embedding predictor, distilled from some other backbone.

    Args:
        random_network - backbone providing target embeddings, RLmodule 
    
    Provides: curiosity
    """
    def __init__(self, network, random_network):
        super().__init__(network=network)
        self.random_network = Reference(random_network)

    def _initialize(self):
        self.random_network.initialize()
        self._output_representation = Embedding(self.random_network.output_representation.embedding_size)
        super()._initialize()

    def curiosity(self, storage):
        prediction = self(storage)
        with torch.no_grad():
            truth = self.random_network(storage)
        return prediction.compare(truth)

    #TODO: __repr__