from LegoRL.core.RLmodule import RLmodule
from LegoRL.samplers.sampler import Sampler

import numpy as np

class SumTree():
    """
    Stores the priorities in sum-tree structure for effecient sampling.
    Tree structure and array storage:
    Tree index:
         0         -> storing priority sum
        / \
      1     2
     / \   / \
    3   4 5   6    -> storing priority for transitions
    Array type for storing:
    [0,1,2,3,4,5,6]
    """

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------parent nodes-------------][-------leaves to record priority-------]
        #             size: capacity - 1                       size: capacity

    def update(self, idx, p):
        """
        input: idx - int, id of leaf to update
        input: p - float, new priority value
        """
        assert idx < self.capacity, "SumTree overflow"
        
        idx += self.capacity - 1  # going to leaf №i
        
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:    # faster than the recursive loop
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, v):
        """
        input: v - float, cumulative priority of first i leafs
        output: i - int, selected index
        """
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx] or self.tree[cr_idx] == 0.0:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        return leaf_idx - (self.capacity - 1)
        
    def __getitem__(self, indices):
        return self.tree[indices + self.capacity - 1]

    @property
    def total_p(self):
        return self.tree[0]  # the root is sum of all priorities

class PrioritizedSampler(Sampler):
    """
    Prioritized sampler.
    Based on: https://arxiv.org/abs/1511.05952

    Args:
        clip_priorities - float or None, clipping priorities as suggested in original paper
        rp_alpha - smoothing of priorities

    Provides: sample, update_priorities
    """
    def __init__(self, par, replay, clip_priorities=1, rp_alpha=0.6, *args, **kwargs):
        super().__init__(par, replay, *args, **kwargs)
        
        self._previous_buffer_pos = 0        
        self.max_priority = 1.0
        self.clip_priorities = clip_priorities
        self.rp_alpha = rp_alpha
        self.priorities = SumTree(self.replay.capacity)

    def sample(self):
        '''
        Samples batch using priorities.
        output: Storage
        '''
        if len(self.replay) < self.cold_start:
            return None

        # sample batch_size indices
        batch_indices = np.array([self.priorities.get_leaf(np.random.uniform(0, self.priorities.total_p)) for _ in range(self.batch_size)])
        
        # get transitions with these indices
        self._sample = self.replay.at(batch_indices)
        self._sample.priorities = self.priorities[batch_indices]
        self._sample.indices = batch_indices
        return self._sample

    def expand(self, indices):
        '''
        Expand priorities with max priority
        input: indices - numpy array, int
        '''
        for idx in indices:
            self.priorities.update(idx, self.max_priority)

    def update_priorities(self, indices, new_priorities):
        '''
        Updates priorities with batch_priorities for transition on indices from current sample.
        input: indices - numpy array, int
        input: new_priorities - Loss
        '''
        new_priorities = (new_priorities.numpy ** self.rp_alpha).clip(min=1e-5, max=self.clip_priorities)
        for i, v in zip(indices, new_priorities):
            self.priorities.update(i, v)
        
        # update max priority for new transitions
        self.max_priority = max(self.max_priority, new_priorities.max())

    def hyperparameters(self):
        return {"clip_priorities": self.clip_priorities, "rp_alpha": self.rp_alpha}
        
    def __repr__(self):
        return f"Samples mini-batch from <{self.replay.name}> using priorities"