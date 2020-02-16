from LegoRL.representations.representation import Which
from LegoRL.representations.policy import Policy
from LegoRL.representations.standard import Embedding
from LegoRL.transformations.head import Head

import torch

class InverseDynamics(Head):
    """
    Inverse dynamics task is to predict action by state and next state.
    Provides: curiosity
    """

    def __init__(self, *args, **kwargs):
        super().__init__(representation=Policy, *args, **kwargs)

    @property
    def input_representation(self):
        return self.mdp[Embedding(self.backbone.output_representation.embedding_size * 2)]

    def _get_input(self, storage, which):
        output = self.backbone(storage, Which.all)
        return torch.cat([output.crop(Which.current).tensor, 
                          output.crop(Which.next).tensor], dim="features")

    def curiosity(self, storage):
        action_predictions = self(storage)
        # NamedTensor issue
        return self.mdp["Loss"](-action_predictions.log_prob(storage.actions.tensor.rename(None)))