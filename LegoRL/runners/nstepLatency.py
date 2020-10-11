from LegoRL.core.RLmodule import RLmodule

from copy import copy

class NstepLatency(RLmodule): 
    """
    Stores transitions more than on one step.
    
    Args:
        n_steps - N steps, int

    Provides: sample
    """       
    def __init__(self, system, n_steps=3):
        super().__init__(system)
        
        self.n_steps = n_steps
        assert self.n_steps > 1

        self.nstep_buffer = []

    def add(self, storage):
        '''
        Stores observation in buffer and pops n-step transition as observation
        input: Storage
        output: Storage
        '''
        self.nstep_buffer.append(copy(storage))

        for i in range(len(self.nstep_buffer) - 1):
            # this was a place of several knee-shots
            self.nstep_buffer[i].rewrite("next_states", storage.next_states)
            self.nstep_buffer[i].rewrite("rewards", self.nstep_buffer[i].rewards + storage.rewards * self.nstep_buffer[i].discounts)
            self.nstep_buffer[i].rewrite("discounts", self.nstep_buffer[i].discounts * storage.discounts)
        
        if len(self.nstep_buffer) == self.n_steps:            
            return self.nstep_buffer.pop(0)
        return None

    def hyperparameters(self):
        return {"n_steps": self.n_steps}

    def __repr__(self):
        return f"Substitutes stream to {self.n_steps}-step transitions"