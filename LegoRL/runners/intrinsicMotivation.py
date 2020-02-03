from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference
from LegoRL.core.reference import ReferenceList

import numpy as np

class IntrinsicMotivation(RLmodule): 
    """
    Adds intrinsic motivation to the rewards from other runner.
    
    Args:
        runner - RLmodule with "sample" method.
        motivations - list of RLmodules with "curiosity" method.
        coeffs - weight of intrinsic reward, float.

    Provides:
        sample - returns Storage with transitions from interaction.
    """       
    def __init__(self, runner, motivations, coeffs=None, frozen=False):
        super().__init__(frozen=frozen)
        
        self.runner = Reference(runner)
        self.motivations = ReferenceList(motivations)
        self.coeffs = coeffs or [1.] * len(self.motivations)
        
        assert len(self.coeffs) == len(self.motivations), "Error: there must be weights per each motivation"
        assert len(self.motivations) > 0, "Error: there must be at least one motivation"

        self._last_seen_id = None
        self.intrinsic_R = [0]*len(self.motivations)
        
    def _iteration(self):
        self.sample(existed=False)

    def sample(self, existed=True):
        '''
        Adds intrinsic motivation to samples from runner.
        output: Storage
        '''
        if self._performed:
            self.debug("returns same sample.")
            return self._sample
        if existed: return None
        self._performed = True
        
        storage = self.runner.sample(existed=False)
        if storage is None:
            self.debug("no new observations found, no observation generated.")
            self._sample = None
            return self._sample

        if self.frozen:
            self.debug("frozen; does nothing.")
            self._sample = storage
            return self._sample

        # Motivation collection. 
        self.debug("starts computing intrinsic motivation.", open=True)
        intrinsic_rewards = 0
        for i in range(len(self.motivations)):
            ir = self.coeffs[i] * self.motivations[i].curiosity(storage).numpy
            intrinsic_rewards += ir

            # logging intrinsic rewards
            if storage.id - 1 != self._last_seen_id:
                self.intrinsic_R[i] = np.zeros((storage.batch_size), dtype=np.float32)
            self.intrinsic_R[i] += ir
            self._last_seen_id = storage.id

            for res in self.intrinsic_R[i][storage.discounts.numpy == 0]:
                self.log(motivation.name + " curiosity", res, "reward", "episode", self.runner.episodes_done)
            self.intrinsic_R[i][storage.discounts.numpy == 0] = 0

        # returning updated transitions
        self.debug("adds intrinsic motivation to transitions.", close=True)
        self._sample = storage.copy()
        self._sample.rewards.numpy += intrinsic_rewards
        return self._sample

    def __repr__(self):
        return f"Adds intrinsic rewards from <{[motivation.name for motivation in self.motivations]}> with weights {self.coeffs} to stream from <{self.runner.name}>"