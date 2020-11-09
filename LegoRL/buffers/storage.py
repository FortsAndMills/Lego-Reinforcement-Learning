from LegoRL.representations.representation import Representation

import numpy as np
from LegoRL.utils.namedTensorsUtils import torch_stack

def stack(to_stack):
    '''
    Stacks objects in list along the first dimension
    input: to_stack - list of either Representations, numpy arrays or scalars. 
    output: numpy array or Representation
    '''
    # list of representations case
    if isinstance(to_stack[0], Representation):
        if hasattr(to_stack, "tensor"):
            data = torch_stack([r.tensor for r in to_stack], 0, "timesteps")
        else:
            data = np.stack([r.numpy for r in to_stack])
        return type(to_stack[0])(data)
    
    # list of numpy arrays case
    if isinstance(to_stack[0], np.ndarray):
        return np.stack(to_stack)
    
    # list of floats case
    return np.array(to_stack)

class Storage(dict):
    '''
    Dictionary of representations.
    '''
    @classmethod
    def from_list(cls, storages):
        '''
        Stacks all tensors in list of storages along "timesteps" axis
        input: storages - list of Storage
        output: Storage
        '''
        return Storage({key: stack([storage[key] for storage in storages]) 
                        for key in storages[0].keys()})

    def types(self):
        '''
        Returns dict of which types are used in this storage
        output: dict <str, cls>
        '''
        return {name: type(repr) for name, repr in self.items()}

    def transitions(self):
        '''
        Iterates over single transitions in this storage
        yield: Storage
        '''
        for data in zip(*self.values()):
            yield {name: repr for name, repr in zip(self.keys(), data)}

    def batch(self, indices):
        '''
        input: indices - list of ints
        output: Storage
        '''
        return Storage({key: data.batch(indices) for key, data in self.items()})

    # def rename(self, key1, key2):
    #     '''
    #     Prevents from modifying keys of storage
    #     '''
    #     assert key1 in self.keys(), "Error: key not found"
    #     assert key2 not in self.keys(), "Error: No, please don't do this"
    #     super().__setitem__(key2, self[key1])
    #     del self[key1]

    # interface----------------------------------------------------------------
    def remove_from_gpu(self):
        for data in self.values():
            data.remove_from_gpu()

    def total_size(self):
        '''
        Returns number of transitions stored
        output: int
        '''
        return next(iter(self.values())).total_size

    def __getattr__(self, attr):
        '''
        Allows access to items like to attributes
        '''
        if attr in self.keys():
            return self[attr]
        raise AttributeError()

    def __setattr__(self, attr, val):
        '''
        Allows adding items like attributes
        '''
        self[attr] = val

    def __getitem__(self, attrs):
        '''
        Allows access to sub storage if a tuple of keys is given as input key
        output then is Storage
        '''
        if isinstance(attrs, tuple):
            return Storage({name: self[name] for name in attrs})
        return super().__getitem__(attrs)

    def __setitem__(self, attr, val):
        '''
        Prevents from modifying keys of storage
        '''
        assert not attr in self.keys(), "Error: no, please don't do this"
        super().__setitem__(attr, val)

    def rewrite(self, attr, val):
        '''
        If you want to update value in storage, explicitly call this function
        '''
        assert attr in self.keys()
        super().__setitem__(attr, val)