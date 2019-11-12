from .sampler import *

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
        replay - RLmodule with "buffer_pos", "buffer", "capacity" properties
        clip_priorities - float or None, clipping priorities as suggested in original paper

    Provides: sample, update_priorities
    """
    def __init__(self, clip_priorities=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.previous_buffer_pos = 0        
        self.max_priority = 1.0
        self.clip_priorities = clip_priorities

    def initialize(self):        
        self.priorities = SumTree(self.replay.capacity)

    def iteration(self):
        # new transitions are stored with max priority
        while self.previous_buffer_pos != self.replay.buffer_pos:
            self.debug("found new element in replay, stored with max priority")
            self.priorities.update(self.previous_buffer_pos, self.max_priority)
            self.previous_buffer_pos += 1

        # then sample a batch as in usual sampler
        super().iteration()

    def generate_sample(self):
        '''
        Samples batch using priorities and demands new priorities for this batch.
        '''
        self.debug("samples new batch using priorities.")

        # sample batch_size indices
        batch_indices = np.array([self.priorities.get_leaf(np.random.uniform(0, self.priorities.total_p)) for _ in range(self.batch_size)])
        
        # get transitions with these indices
        # seems like the fastest code for sampling!
        samples = [self.replay.buffer[idx] for idx in batch_indices]        
        
        self._sample = self.system.Batch(*zip(*samples), self.replay.n_steps)
        self._sample.priorities = self.priorities[batch_indices]
        self._sample.indices = batch_indices

    def update_priorities(self, batch_priorities):
        '''
        Updates priorities with batch_priorities for transition on indices from current sample.
        input: batch_priorities - numpy array, (*batch_shape)
        '''
        new_batch_priorities = batch_priorities.clip(min=1e-5, max=self.clip_priorities)
        for i, v in zip(self._sample.indices, new_batch_priorities):
            self.priorities.update(i, v)
        
        # update max priority for new transitions
        self.max_priority = max(self.max_priority, new_batch_priorities.max())
        
    def __repr__(self):
        return f"Each {self.timer} iteration samples mini-batch from {self.replay.name} using priorities"