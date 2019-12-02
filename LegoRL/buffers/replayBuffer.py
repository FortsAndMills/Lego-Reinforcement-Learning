from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.composed import Reference

import psutil

class ReplayBuffer(RLmodule):
    """
    Replay Memory storing all transitions from runner.
    
    Args:
        runner - RLmodule with "new_transitions" and "was_reset" properties
        capacity - size of buffer, int

    Provides: buffer, buffer_pos, __len__, capacity
    """
    def __init__(self, runner, capacity=10000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.runner = Reference(runner)
        self.capacity = capacity
        
        self.buffer = []
        self.buffer_pos = 0
    
    def _store_transition(self, transition):
        """
        Remembers given transition.
        input: Transition
        """
        # this seems to be the quickest way of working with experience memory
        if len(self) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.buffer_pos] = transition
        
        self.buffer_pos = (self.buffer_pos + 1) % self.capacity
        
        # check if there is enough virtual memory
        if len(self) == 1:
            assert self.buffer[0].size() * self.capacity < psutil.virtual_memory().available
    
    def iteration(self):
        """
        Collects observations from runner.
        """   
        batch = self.runner.new_transitions()
        if batch is not None:
            self.debug("adds new observations from runner.")

            # TODO: need some good idea how to share reference between next_state
            # and state of following transitions coming from same rollout.
            # Untrivial cause runner can be with latency.            
            for transition in batch.transitions():                
                self._store_transition(transition)
        else:
            self.debug("no new observations found.")
        
    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return f"Stores observations from <{self.runner.name}>"