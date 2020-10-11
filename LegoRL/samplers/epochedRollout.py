from LegoRL.core.RLmodule import RLmodule

import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class EpochedRollout(RLmodule):
    """
    Performs several epochs through collected rollout.
    Used for "reusing" samples in policy gradients algorithms.
    Based on: https://arxiv.org/abs/1707.06347
    
    Args:
        epochs - number of epochs to run through rollout on each update
        batch_size - size of mini-batch to select without replacement on each gradient ascent step   

    Provides: sample
    """
    def __init__(self, sys, epochs = 3, batch_size=32):
        super().__init__(sys)

        self.epochs = epochs
        self.batch_size = batch_size

        self._sampler_idx = 0
        self._epochs_cnt = 0
        self.data = None
        self._sampler = None

    def new_dataset(self, storage):
        '''
        Sets new data
        input: Storage
        '''
        assert self.data is None
        self.data = storage
        self._epochs_cnt = 0
        self._sampler = None

    def sample_next(self):
        '''
        output: Storage or NOne
        '''
        if self.data is None:
            return None

        # if we do not have sampler yet
        if self._sampler is None:
            self._sampler_idx = 0
            self._sampler = list(BatchSampler(SubsetRandomSampler(range(self.data.total_size())), self.batch_size, drop_last=True))
                
        # next batch
        indices = np.array(self._sampler[self._sampler_idx])
        self._sampler_idx += 1
        
        sample = self.data.batch(indices)

        # if sampler ended, epoch is finished
        if self._sampler_idx >= len(self._sampler):
            self._sampler = None
            self._epochs_cnt += 1

            # if epochs ended, rollout is finished
            if self._epochs_cnt == self.epochs:
                self._epochs_cnt = 0
                self.data = None

        return sample

    def hyperparameters(self):
        return {"epochs": self.epochs, "batch_size": self.batch_size}

    def __repr__(self):
        return f"Performs {self.epochs} epoches with batches of size {self.batch_size} on provided dataset"