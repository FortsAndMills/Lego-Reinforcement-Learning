from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference

import numpy as np
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

        self.rollout_sample = None
        self._sampler = None
        self._sampler_idx = 0
        self._epochs_cnt = 0

    def sample(self, trigger=True):
        """
        Samples mini-batches from collected rollouts.
        input: trigger - if False, sample will be returned only if it already exists
        output: Storage
        """
        assert trigger
        if self._performed:
            self.debug("returns same sample.")
            return self._sample
        self._performed = True

        # if we do not have rollout yet
        if self.rollout_sample is None:
            self.rollout_sample = self.rollout.sample()
        
        if self.rollout_sample is None:
            self.debug("new rollout is not generated yet.")
            self._sample = None
            return self._sample

        # if we do not have sampler yet
        if self._sampler is None:
            self._sampler_idx = 0
            self._sampler = list(BatchSampler(SubsetRandomSampler(range(len(self.rollout_sample))), self.batch_size, drop_last=False))
        
        # if sampler ended, epoch is finished
        if self._sampler_idx >= len(self._sampler):
            self.debug("epoch has ended.")
            self._sampler = None
            self._epochs_cnt += 1

            # if epochs ended, rollout is finished
            if self._epochs_cnt == self.epochs:
                self.debug("this rollout is finished, new one is required!")
                self._epochs_cnt = 0
                self.rollout_sample = None
            
            self._performed = False
            return self.sample()
        
        # next batch
        indices = np.array(self._sampler[self._sampler_idx])
        self._sampler_idx += 1

        # creating storage
        self._sample = self.rollout_sample.batch(indices)
        self._sample.indices = indices
        self._sample.full_rollout = self.rollout_sample
        return self._sample

    # # TODO: wtf? and what if I want different targets for different losses?
    # def returns(self, sample):
    #     return self.target.returns(self.rollout_sample)[sample.indices]
    
    # def advantage(self, sample):
    #     return self.target.advantage(self.rollout_sample)[sample.indices]

    def hyperparameters(self):
        return {"epochs": self.epochs, "batch_size": self.batch_size}

    def __repr__(self):
        return f"Performs {self.epochs} epoches with batches of size {self.batch_size} using rollouts from <{self.rollout.name}>"