from .RLmodule import *

class NstepLatency(RLmodule): 
    """
    Stores transitions more than on one step.
    
    Args:
        runner - RLmodule with "observation" and "was_reset" properties
        n_steps - N steps, int

    Provides: observation, was_reset
    """       
    def __init__(self, runner, n_steps=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.runner = Reference(runner)
        self.n_steps = n_steps
        self.nstep_buffer = []

        self._was_reset = None
        self._observation = None

    @property
    def was_reset(self):
        return self._was_reset
    
    @property
    def observation(self):
        return self._observation

    def iteration(self):
        '''Stores observation in buffer and pops n-step transition as observation'''
        transitionBatch = self.runner.observation
        if transitionBatch is None:
            self.debug("no new observations found, no observation generated")
            self._was_reset = None
            self._observation = None
            return

        self._was_reset = self.runner.was_reset
        if self.runner.was_reset:
            self.debug("runner was reset, so I reset too (nstep buffer cleared)")
            self.nstep_buffer = []

        self.debug("adds new observations from runner")
        self.nstep_buffer.append(transitionBatch)
        
        if len(self.nstep_buffer) == self.n_steps:      
            nstep_reward = sum([self.nstep_buffer[i].reward * (self.system.gamma**i) for i in range(self.n_steps)])
            actual_done = max([self.nstep_buffer[i].done for i in range(self.n_steps)])
            
            oldestTransitions = self.nstep_buffer.pop(0)
            state, action = oldestTransitions.state, oldestTransitions.action
            
            self._observation = TransitionBatch(state, action, nstep_reward, transitionBatch.next_state, actual_done)
            self._observation.n_steps = self.n_steps
        else:
            self._was_reset = None
            self._observation = None            

        if len(self.nstep_buffer) >= self.n_steps:
            raise Exception("Error! Nstep buffer is >N")

    def wait(self):
        '''
        Still builds nstep_buffer (consider example with timer = 2)
        but does not stream observations if frozen during iteration.
        '''
        self.iteration()
        self._was_reset = None
        self._observation = None

        # TODO: may be, if frozen, it shouldn't modify stream?

    def __repr__(self):
        return f"Substitutes stream from {self.runner.name} to {self.n_steps}-step transitions"

