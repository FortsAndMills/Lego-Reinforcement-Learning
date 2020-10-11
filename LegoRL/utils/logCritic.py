from LegoRL.models.model import Model

import torch

class LogCritic():
    """
    Logs critic value of initial state

    Args:
        critic - callable, returning V by states
    """
    def __init__(self, critic):
        self.critic = critic

    def visualize(self):
        '''Stores value of initial state.'''
        if self.system.time_for_rare_logs():
            v0 = self.critic(self.system.initial_state_example)
            v0 = v0.scalar().detach().tensor.item()
            self.log(self.name + " initial state value", v0, "$V(s_0)$")