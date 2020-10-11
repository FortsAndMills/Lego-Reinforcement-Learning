from LegoRL.core.RLmodule import RLmodule
from LegoRL.runners.interactor import Interactor

import os
import time
import pickle
import numpy as np
from itertools import count

class Runner(Interactor):
    """
    Performs interaction step-by-step with logging of results.

    Args:
        log_info - which values from info to log, list of tuples (key, y_axis):
            key - name of info to log, str
            y_axis - name of y-axis for this log, str or None

    Provides:
        step - returns Storage with transitions from one step of interaction.
    """
    def __init__(self, *args, log_info=[], **kwargs):
        super().__init__(*args, **kwargs)

        self.log_info = log_info
        self.frames_done = 0
        self.episodes_done = 0

    def step(self, actions):
        """
        Makes one step and logs results.
        input: actions - Action
        input: keys - list of str
        output: Storage
        """
        start = time.time()
        storage = self._perform_step(actions)
        self.log("playing time", time.time() - start, "seconds")
        
        self.frames_done += self.env.num_envs
        for i in range(self.env.num_envs):
            if self.done[i]:
                self.episodes_done += 1

                self.log("rewards", self.R.numpy[i], "reward")
                self.log("lengths", self.T[i], "length")
                self.log("episode ends", self.frames_done)

                for key, yaxis in self.log_info:
                    self.log(key, self.info[i][key], yaxis)
        return storage

    def hyperparameters(self):
        return {"num_envs": self.env.num_envs}

    @property
    def fps(self):
        """
        Returns fps for this runner.
        output: float
        """
        if self.system.wallclock() == 0: return "Unknown"
        return self.frames_done / self.system.wallclock()

    # interface functions ----------------------------------------------------------------
    def _save(self, folder_name):        
        with open(os.path.join(folder_name, self.name), 'wb') as f:
            pickle.dump(self.frames_done, f)
            pickle.dump(self.episodes_done, f)
        
    def _load(self, folder_name):     
        with open(os.path.join(folder_name, self.name), 'rb') as f:
            self.frames_done = pickle.load(f)
            self.episodes_done = pickle.load(f)

    def __repr__(self):
        return f"Makes steps in {self.env.num_envs} parallel environments"
