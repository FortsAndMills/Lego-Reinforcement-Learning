from LegoRL.core.RLmodule import RLmodule
from LegoRL.transformations.head import Head

import torch

class CriticHead(Head):
    """
    Provides a head for critics.

    Args:
        log_init_evaluation - whether to log output of critic on s_0, bool
    """
    def __init__(self, log_init_evaluation=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.log_init_evaluation = log_init_evaluation

    def _visualize(self):
        '''Stores value of initial state.'''
        if self.log_init_evaluation and self.system.time_for_rare_logs():
            # TODO: cache issue?
            v0 = self(self.system.initial_state_example).scalar().detach().item()
            self.log(self.name + " initial state value", v0, "$V(s_0)$")

    def hyperparameters(self):
        return {"representation": self._output_representation._default_name()}