from LegoRL.core.RLmodule import RLmodule
from LegoRL.buffers.storage import Storage

import numpy as np
from collections import defaultdict

class ReplayBuffer(RLmodule):
    """
    Replay Memory storing data in raw numpy format.
    Used for experience replay.
    
    Args:
        capacity - size of buffer, int

    Provides:
        store - add data from Storage to buffer
        at - get data by indices
    """
    def __init__(self, par, capacity=10000):
        super().__init__(par)
        
        self.capacity = capacity        
        self._buffer = defaultdict(list)
        self._buffer_pos = 0
        self._size = 0
        self._types = None
    
    def _store_transition(self, storage):
        """
        Remembers single transition.
        input: Storage
        output: index of storing, int
        """
        if self._size < self.capacity:
            self._size += 1
            for name, data in storage.items():
                self._buffer[name].append(data.numpy)
        else:
            for name, data in storage.items():
                self._buffer[name][self._buffer_pos] = data.numpy
        
        idx = self._buffer_pos
        self._buffer_pos = (self._buffer_pos + 1) % self.capacity
        return idx
        
    def store(self, storage):
        """
        Remembers given transitions.
        input: Storage
        output: index of storing, list of ints
        """
        if self._types is None:
            self._types = storage.types()
        else:
            assert self._types == storage.types(), f"Error: Replay Buffer expected scheme {self._types}; received scheme {storage.types()}"

        idxs = []
        for transition in storage.transitions():              
            idxs.append(self._store_transition(transition))
        return idxs

    def at(self, indices):
        """
        Returns storage with data on given indices
        input: indices - list of ints
        output: Storage
        """
        return Storage({
            name: ty(np.stack([self._buffer[name][i] for i in indices])) 
            for name, ty in self._types.items()
        })

    def __len__(self):
        return self._size

    def hyperparameters(self):
        return {"capacity": self.capacity}        

    def __repr__(self):
        return f"Stores data in raw numpy format"