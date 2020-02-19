from LegoRL.core.RLmodule import RLmodule
from LegoRL.representations.representation import Which
from LegoRL.core.reference import Reference
from LegoRL.transformations.head import Head
from LegoRL.representations.standard import Embedding

import torch
import numpy as np

class ForwardDynamics(Head):
    """
    Decoder trying to predict next state in some embedding space.
    Takes as input embedding of state and action.
    
    Provides: curiosity
    """
    @property
    def input_representation(self):
        return self.mdp[Embedding(self.backbone.output_representation.rshape().numel() + 
                                  np.prod(self.mdp.action_description_shape))]

    @property
    def output_representation(self):
        return self.backbone.output_representation

    def _get_input(self, storage, which):
        states = self.backbone(storage).tensor
        states = states.rename(**{states.names[-1]: "features"})
        actions = self.mdp.action_preprocessing(storage.actions.tensor)
        return torch.cat([states, actions], dim="features")

    def curiosity(self, storage):
        prediction = self(storage)
        truth = self.backbone(storage, Which.next)
        return prediction.compare(truth)

    def __repr__(self):
        connected = f", using <{self.backbone.name}> as state embeddings." if isinstance(self.backbone, RLmodule) else "" 
        return f"Predicts next state representation from current and action" + connected