from LegoRL.core.RLmodule import RLmodule
from LegoRL.runners.interactor import Interactor

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

    Provides:
        sample - returns Storage with transitions from interaction.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.frames_done = 0
        self.episodes_done = 0
        self._sample = None

    def sample(self, existed=True):
        """
        Makes one step and logs results.
        input: existed - bool, if True transitions will be returned only if they exist already
        output: Storage
        """
        if self._performed:
            self.debug("returns same sample.")
            return self._sample
        if existed: return None
        self._performed = True

        self.debug("plays one step.", open=True)

        start = time.time()
        self._sample, results, lengths = self.step()
        self.log("playing time", time.time() - start, "seconds")

        self.debug(close=True)
        
        self.frames_done += self.env.num_envs
        for res, leng in zip(results, lengths):
            self.episodes_done += 1

            self.log("rewards", res, "reward", "episode", self.episodes_done)
            self.log("lengths", leng, "length", "episode", self.episodes_done)
            self.log("episode ends", self.frames_done)

        return self._sample

    def _iteration(self):
        # triggers one step in environment.
        self.sample(existed=False)

    @property
    def fps(self):
        """
        Returns fps for this runner.
        output: float
        """
        return self.frames_done / self.system.wallclock()

    # interface functions ----------------------------------------------------------------
    def _write(self, f):
        pickle.dump(self.frames_done, f)
        pickle.dump(self.episodes_done, f)
        
    def _read(self, f):
        self.frames_done = pickle.load(f)
        self.episodes_done = pickle.load(f)

    def __repr__(self):
        return f"Makes one step in {self._threads} parallel environments each {self.timer} iteration using <{self.policy.name}>"
