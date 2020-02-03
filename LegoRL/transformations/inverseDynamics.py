from LegoRL.transformations.head import Head
from LegoRL.representations.policy import Policy

import torch

class InverseDynamics(Head):
    """
    Inverse dynamics task is to predict action by state and next state.
    Provides: curiosity
    """

    def __init__(self, *args, **kwargs):
        super().__init__(representation=Policy, *args, **kwargs)

    def curiosity(self, storage):
        action_predictions = self(storage)
        # NamedTensor issue
        return -action_predictions.log_prob(storage.actions.rename(None))