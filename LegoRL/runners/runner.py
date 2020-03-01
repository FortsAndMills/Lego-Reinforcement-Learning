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
    By default, performs one step each iteration.
    This can be modified using timer parameter or turned off by setting frozen=False.
    The last will lead to performing steps only when required by other modules.

    Args:
        log_info - which values from info to log, list of tuples (key, y_axis):
            key - name of info to log, str
            y_axis - name of y-axis for this log, str or None

    Provides:
        sample - returns Storage with transitions from interaction.
    """
    def __init__(self, log_info=[], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log_info = log_info

        self.frames_done = 0
        self.episodes_done = 0
        self._sample = None

    def sample(self, trigger=False):
        """
        Makes one step and logs results.
        input: trigger - if False, sample will be returned only if it already exists
        output: Storage
        """
        if self._performed:
            self.debug("returns same sample.")
            return self._sample
        if not trigger: return None
        self._performed = True

        self.debug("plays one step.", open=True)

        start = time.time()
        self._sample, info = self.step()
        self.log("playing time", time.time() - start, "seconds")

        self.debug(close=True)
        
        self.frames_done += self.env.num_envs
        for i in range(self.env.num_envs):
            if self.done[i]:
                self.episodes_done += 1

                self.log("rewards", self.R[i], "reward", "episode", self.episodes_done)
                self.log("lengths", self.T[i], "length", "episode", self.episodes_done)
                self.log("episode ends", self.frames_done)

                for key, yaxis in self.log_info:
                    self.log(key, info[i][key], yaxis, "episode", self.episodes_done)

        return self._sample

    def _iteration(self):
        # triggers one step in environment.
        self.sample(trigger=True)

    def hyperparameters(self):
        return {"timer": self.timer,
                "num_envs": self.env.num_envs if self._initialized else f"{self._threads} threads"
        }

    @property
    def fps(self):
        """
        Returns fps for this runner.
        output: float
        """
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
        return f"Makes one step in {self._threads} parallel environments each {self.timer} iteration using <{self.policy.name}> policy"
