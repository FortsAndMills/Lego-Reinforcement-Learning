from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class EpochedRollout(RLmodule):
    """
    Performs several epochs through collected rollout.
    Used for "reusing" samples in policy gradients algorithms.
    Based on: https://arxiv.org/abs/1707.06347
    
    Args:
        rollout - RLmodule with "sample" method
        epochs - number of epochs to run through rollout on each update
        batch_size - size of mini-batch to select without replacement on each gradient ascent step   

    Provides: sample
    """
    def __init__(self, rollout, epochs = 3, batch_size=32):
        super().__init__()

        self.rollout = Reference(rollout)
        self.epochs = epochs
        self.batch_size = batch_size

    def sample(self):
        """
        Samples mini-batches from collected rollouts.
        output: Batch
        """
        if self._performed:
            self.debug("returns same sample.")
            return self._sample
        self._performed = True

        if self.rollout_sample is None:
            self.rollout_sample = self.rollout.sample(existed=False)
        
        if self.rollout_sample is None:
            self.debug("new rollout is not generated yet.")
            self._sample = None
            return self._sample

        if self._sampler is None:
            self._sampler = BatchSampler(SubsetRandomSampler(range(len(self.rollout_sample))), self.batch_size, drop_last=False)
            
        indicies = next(self._sampler)

        if indicies is None:
            self.debug("epoch has ended.")
            self._sampler = None
            self._epochs_cnt += 1

            if self._epochs_cnt == self.epochs:
                self.debug("this rollout is finished, new one is required!")
                self._epochs_cnt = 0
                self.rollout_sample = None

            return self.sample()

        self._sample = self.rollout_sample[indices]
        return self._sample

    def __repr__(self):
        return f"Collects rollouts of length {self.rollout_length} from <{self.runner.name}>"