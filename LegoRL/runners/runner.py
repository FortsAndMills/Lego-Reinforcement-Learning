from LegoRL.runners.interactor import Interactor

import time
import pickle
import numpy as np
from itertools import count

class Runner(Interactor):
    """
    Performs interaction step-by-step with logging of results.
    By default, performts one step each iteration.

    Provides: transitions, new_transitions
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.frames_done = 0
        self.episodes_done = 0
        self._transitions = None
    
    def new_transitions(self):
        return self._transitions if self.performed else None

    def transitions(self):
        """
        Makes one step and logs results.
        Output: Batch
        """
        if self.performed:
            self.debug("returns same transitions")
            return self._transitions
        self.performed = True

        self.debug("plays one step.", open=True)

        start = time.time()
        self._transitions, results = self.step()
        self.log("playing time", time.time() - start, "training iteration", "seconds")
        
        self.frames_done += self.env.num_envs
        for res in results:
            self.episodes_done += 1

            self.log("rewards", res, "episode", "reward", self.episodes_done)
            self.log("episode ends", self.frames_done)
            # TODO: log episode length
            # TODO: fps

        self.debug(close=True)

        return self._transitions

    def iteration(self):
        self.transitions()

    def _write(self, f):
        pickle.dump(self.frames_done, f)
        pickle.dump(self.episodes_done, f)
        
    def _read(self, f):
        self.frames_done = pickle.load(f)
        self.episodes_done = pickle.load(f)

    def __repr__(self):
        # TODO: bring back {self.env.num_envs}
        return f"Makes one step in parallel environments each {self.timer} iteration using <{self.policy.name}>"