from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.composed import Reference

class ReplayBuffer(RLmodule):
    """
    Replay Memory storing all transitions from runner.
    Based on: https://arxiv.org/abs/1312.5602
    
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
    
    def iteration(self):
        """
        Collects observations from runner.
        """   
        batch = self.runner.new_transitions()
        if batch is not None:
            self.debug("adds new observations from runner")
            
            # TODO: only for n_steps = 1!!!
            for transition in batch.transitions():
            #     # if runner was not reset, we just store link to the same object as next_state for previous transition
            #     # this should reduce memory consumption twice
            #     if not self.runner.was_reset:
            #         transition.state = self.buffer[self.buffer_pos - batch.batch_size].next_state
                
                self._store_transition(transition)
        else:
            self.debug("no new observations found")
        
    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return f"Stores observations from <{self.runner.name}>"