from LegoRL.core.RLmodule import RLmodule
from LegoRL.buffers.storage import Storage

class RolloutCollector(RLmodule):
    """
    Collects rollouts of given length from runner.
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        rollout_length - length of rollout to collect on each iteration, int        

    Provides: sample
    """
    def __init__(self, sys, rollout_length = 5):
        super().__init__(sys)

        self.rollout_length = rollout_length
        self._rollout = []

    def add(self, storage):
        """
        Adds transitions from runner to rollout and creates a sample if desired length is reached.
        """
        self._rollout.append(storage)        
        assert len(self._rollout) <= self.rollout_length        
        
        if len(self._rollout) == self.rollout_length:
            dataset = Storage.from_list(self._rollout)
            self._rollout = []
            return dataset
        
        return None

    def hyperparameters(self):
        return {"rollout_length": self.rollout_length}

    def __repr__(self):
        return f"Collects rollouts of length {self.rollout_length}"