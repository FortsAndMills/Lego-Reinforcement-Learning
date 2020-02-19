from LegoRL.core.RLmodule import RLmodule
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
        return self.mdp[Embedding(self.backbone.output_representation.rshape().numel() * 2)]

    def _get_input(self, storage, which):
        output = self.backbone(storage, Which.all)

        states = output.crop(Which.current).tensor
        states = states.rename(**{states.names[-1]: "features"})

        next_states = output.crop(Which.next).tensor
        next_states = next_states.rename(**{next_states.names[-1]: "features"})

        return torch.cat([states, next_states], dim="features")

    def curiosity(self, storage):
        action_predictions = self(storage)
        # NamedTensor issue
        return self.mdp["Loss"](-action_predictions.log_prob(storage.actions.tensor.rename(None)))

    def __repr__(self):
        connected = f", using state embeddings from <{self.backbone.name}>." if isinstance(self.backbone, RLmodule) else "" 
        return f"Predicts actions happened between state and next state" + connected