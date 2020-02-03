from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference
from LegoRL.buffers.rolloutStorage import RolloutStorage

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

        self._sample = None
        self._last_seen_id = None

    def sample(self):
        """
        Adds transitions from runner to rollout and creates a sample if ?!?.
        output: Batch
        """
        if self._performed:
            self.debug("returns same sample.")
            return self._sample
        self._performed = True

        runner_sample = self.runner.sample(existed=False)
        if runner_sample is None:
            self.debug("no new observations found, resetting rollout.")
            self._sample = None
            self._rollout = []
            return self._sample

        if runner_sample.id - 1 != self._last_seen_id:
            self._rollout = []
            self.debug("resets because runner was reset.")
        self._last_seen_id = runner_sample.id               
        
        self._rollout.append(runner_sample)
        
        assert len(self._rollout) <= self.rollout_length        
        if len(self._rollout) == self.rollout_length:
            self.debug("rollout generated!")
            self._sample = self.mdp[RolloutStorage].from_list(self._rollout)
            self._rollout = []
        else:
            self.debug("not enough transitions collected.")
            self._sample = None

        return self._sample

    def __repr__(self):
        return f"Collects rollouts of length {self.rollout_length} from <{self.runner.name}>"