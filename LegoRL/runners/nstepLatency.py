from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.composed import Reference

class NstepLatency(RLmodule): 
    """
    Stores transitions more than on one step.
    
    Args:
        runner - RLmodule with "new_transitions" and "was_reset" properties
        n_steps - N steps, int

    Provides: transitions, new_transitions, was_reset
    """       
    def __init__(self, runner, n_steps=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.runner = Reference(runner)
        self.n_steps = n_steps
        self.nstep_buffer = []

        self._was_reset = None
        self._transitions = None

    @property
    def was_reset(self):
        return self._was_reset

    def new_transitions(self):
        return self._transitions if self.performed else None

    def iteration(self):
        self.transitions()

    def transitions(self):
        '''
        Stores observation in buffer and pops n-step transition as observation
        output: Batch
        '''
        if self.performed:
            self.debug("returns same transitions")
            return self._transitions
        self.performed = True

        transitionBatch = self.runner.new_transitions()
        if transitionBatch is None:
            self.debug("no new observations found, no observation generated")
            self._was_reset = None
            self._transitions = None
            return self._transitions

        if self.runner.was_reset:
            self.debug("runner was reset, so I reset too (nstep buffer cleared)")
            self.nstep_buffer = []

        self.debug("adds new observations from runner")
        self.nstep_buffer.append((transitionBatch, self.runner.was_reset))

        for i in range(len(self.nstep_buffer) - 1):
            self.nstep_buffer[i][0].next_states = self.nstep_buffer[-1][0].next_states
            self.nstep_buffer[i][0].rewards += self.nstep_buffer[-1][0].rewards * self.nstep_buffer[i][0].discounts
            self.nstep_buffer[i][0].discounts *= self.nstep_buffer[-1][0].discounts
        
        if len(self.nstep_buffer) == self.n_steps:            
            self._transitions, self._was_reset = self.nstep_buffer.pop(0)
        else:
            self._was_reset = None
            self._transitions = None            

        assert len(self.nstep_buffer) < self.n_steps
        return self._transitions

    def __repr__(self):
        return f"Substitutes stream from <{self.runner.name}> to {self.n_steps}-step transitions"