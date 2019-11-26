from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.composed import Reference
from LegoRL.buffers.rollout import Rollout

class RolloutCollector(RLmodule):
    """
    Collects rollouts of given length from runner.
    Based on: https://arxiv.org/abs/1312.5602
    
    Args:
        runner - RLmodule with "transitions" and "was_reset" properties
        rollout_length - length of rollout to collect on each iteration, int        

    Provides: sample
    """
    def __init__(self, runner, rollout_length = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.runner = Reference(runner)
        self.rollout_length = rollout_length

        self._rollout = Rollout()
        self._sample = None

    def sample(self):
        """
        Adds transitions from runner to rollout and creates a sample if ?!?.
        output: Batch
        """
        if self.performed:
            self.debug("returns same sample")
            return self._sample
        self.performed = True

        transitions = self.runner.transitions()
        if transitions is None:
            self.debug("no new observations found, resetting rollout")
            self._sample = None
            self._rollout = Rollout()
            return self._sample

        if self.runner.was_reset:
            self._rollout = Rollout()
            self.debug("resets because runner was reset")                
        
        self._rollout.append(transitions)
        
        assert self._rollout.rollout_length <= self.rollout_length        
        if self._rollout.rollout_length == self.rollout_length:
            self._sample = self._rollout.to_torch(self.system)
            self._rollout = Rollout()
        else:
            self._sample = None

        return self._sample

    def __repr__(self):
        return f"Collects rollouts of length {self.rollout_length} from <{self.runner.name}>"