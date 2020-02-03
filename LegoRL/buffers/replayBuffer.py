from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference

class ReplayBuffer(RLmodule):
    """
    Replay Memory storing all transitions from runner.
    
    Args:
        runner - RLmodule with "new_transitions", "was_reset" properties
        capacity - size of buffer, int

    Provides:
        buffer - list of Transition
        buffer_pos - int
        __len__ - function, returns size of buffer
        capacity - int
    """
    def __init__(self, runner, capacity=10000, frozen=False):
        super().__init__(frozen=frozen)

        self.runner = Reference(runner)
        self.capacity = capacity
        
        self.buffer = []
        self.buffer_pos = 0
        self._last_seen_id = None
    
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
        Collects new observations from runner.
        """   
        storage = self.runner.sample()
        if storage is not None:
            self.debug("adds new observations from runner.")

            for transition in storage.transitions():
                # saving memory by not storing same state twice
                # we can do that if storage's id is following previous id
                if ((not hasattr(storage, "n_steps") or storage.n_steps == 1) and
                    storage.id - 1 == self._last_seen_id):
                    transition.state = self.buffer[self.buffer_pos - len(storage)].next_state
                self._last_seen_id = storage.id

                # storing transition                
                self._store_transition(transition)
        else:
            self.debug("no new observations found.")
        
    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return f"Stores observations from <{self.runner.name}>"