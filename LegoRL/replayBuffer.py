from .RLmodule import *

class ReplayBuffer(RLmodule):
    """
    Replay Memory storing all transitions from runner.
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        runner - RLmodule with "observation" and "was_reset" properties
        capacity - size of buffer, int

    Provides: buffer, buffer_pos, __len__, capacity
    """
    def __init__(self, runner, capacity=100000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.runner = Reference(runner)
        self.n_steps = None
        self.capacity = capacity
        
        self.buffer = []
        self.buffer_pos = 0
    
    def store_transition(self, transition):
        """
        Remembers given transition.
        input: Transition
        """
        # preparing for concatenation into batch in future
        transition.state      = transition.state[None]
        transition.next_state = transition.next_state[None]
        
        # this seems to be the quickest way of working with experience memory
        if len(self) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.buffer_pos] = transition
        
        self.buffer_pos = (self.buffer_pos + 1) % self.capacity
    
    def iteration(self):
        """
        Collects observations from runner and samples a mini-batch.
        """   
        transitionBatch = self.runner.observation
        if transitionBatch is not None:
            self.debug("adds new observations from runner")

            # updating n_steps of this replay buffer...
            # TODO not very elegant
            n_steps = transitionBatch.n_steps if hasattr(transitionBatch, "n_steps") else 1
            self.n_steps = self.n_steps or n_steps
            assert self.n_steps == n_steps, f"Replay with constant n={self.n_steps} witnessed transitions of length {n_step}"
            
            for transition in transitionBatch:
                self.store_transition(transition)
        else:
            self.debug("no new observations found")
        
    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return f"Stores observations from {self.runner.name}"