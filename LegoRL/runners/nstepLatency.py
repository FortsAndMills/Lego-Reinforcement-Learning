from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference

from copy import copy

class NstepLatency(RLmodule): 
    """
    Stores transitions more than on one step.
    
    Args:
        runner - RLmodule with "sample" method
        n_steps - N steps, int

    Provides: sample
    """       
    def __init__(self, runner, n_steps=3, frozen=False):
        super().__init__(frozen=frozen)
        
        self.runner = Reference(runner)
        self.n_steps = n_steps
        assert self.n_steps > 1

        self.nstep_buffer = []
        self._last_seen_id = None

    def _iteration(self):
        self.sample(existed=False)

    def sample(self, existed=True):
        '''
        Stores observation in buffer and pops n-step transition as observation
        output: Storage
        '''
        if self._performed:
            self.debug("returns same transitions.")
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
            self._last_seen_id = storage.id
            return self._sample

        if storage.id - 1 != self._last_seen_id:
            self.debug("runner was reset, so I reset too (nstep buffer cleared).")
            self.nstep_buffer = []
        self._last_seen_id = storage.id

        self.debug("adds new observations from runner.")
        self.nstep_buffer.append(copy(storage))

        for i in range(len(self.nstep_buffer) - 1):
            self.nstep_buffer[i].next_states.numpy = self.nstep_buffer[-1].next_states.numpy
            self.nstep_buffer[i].rewards.numpy += self.nstep_buffer[-1].rewards.numpy * self.nstep_buffer[i].discounts.numpy
            self.nstep_buffer[i].discounts.numpy *= self.nstep_buffer[-1].discounts.numpy
        
        if len(self.nstep_buffer) == self.n_steps:            
            self._sample = self.nstep_buffer.pop(0)
            self._sample.n_steps = self.n_steps
        else:
            self._sample = None            

        assert len(self.nstep_buffer) < self.n_steps
        return self._sample

    def hyperparameters(self):
        return {"n_steps": self.n_steps}

    def __repr__(self):
        return f"Substitutes stream from <{self.runner.name}> to {self.n_steps}-step transitions"