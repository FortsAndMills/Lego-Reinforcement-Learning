from LegoRL.utils.namedTensorsUtils import torch_unflatten, torch_index, torch_stack, torch_set_index

import torch
import numpy as np

#from enum import Enum
#Which = Enum('Which', 'current next last all')

'''
Representations are tensors for such entities like states, actions, value functions, policies, etc.
They add additional features like:
- transforming Q-function to V-function, 
- handle numpy-PyTorch casts
- shape checking and naming dimensions.

We use PyTorch 1.3.0 unstable Named Tensors feature to simplify aligning procedures.
The motivation lies in using different representations for value functions:
Q-function  - adds "actions" dimension
Categorical - adds "atoms" dimension
MultiReward - adds "rewards" dimension
MultiGamma - adds "discounts" dimension
and they can be combined in complex variations.

Representations are used as inputs and outputs for Model modules in this framework.
'''

class Representation():
    '''
    Base class for representations in this framework.
    
    Args:
        data - Tensor, tuple, list or numpy

    Provides:
        tensor - data in PyTorch Tensor format
        numpy - data in Numpy format
    '''
    def __init__(self, data):
        assert not isinstance(data, Representation)

        # translating to numpy lists and tuples
        if isinstance(data, tuple) or isinstance(data, list) or np.isscalar(data):
            data = np.array(data)
        
        # shape / names checking
        full_name = self._parse_batch_dims(data)
        
        # storing data
        if isinstance(data, np.ndarray):
            self._numpy = data
        else:
            self._tensor = data.refine_names(*full_name)
    
    @classmethod
    def from_linear(cls, tensor):
        '''
        Representation constructor from output of linear layer.
        input: tensor - Tensor, containing unprocessed representation in its last dimension.
        output: Representation
        '''
        assert tensor.shape[-1] == cls.rshape().numel()

        tensor = tensor.refine_names(..., "features")
        tensor = torch_unflatten(tensor, "features", zip(cls.rnames(), cls.rshape()))
        
        return cls(tensor)

    def _parse_batch_dims(self, data):
        '''
        Parses additional batch dimensions in data.
        Stores batch_size and rollout_length in properties.
        input: data - Tensor or numpy array
        output: full tuple of dimension names - tuple of strings
        '''
        assert len(self.rshape()) == 0 or data.shape[-len(self.rshape()):] == self.rshape()

        names = self.rnames()
        extra_dims = len(data.shape) - len(names)
        assert extra_dims <= 2, "ERROR: Weird batch shape"

        if extra_dims == 0:
            self.batch_size = 1
            self.rollout_length = 1
        if extra_dims == 1:
            names = ("batch",) + names
            self.batch_size = data.shape[0]
            self.rollout_length = 1
        if extra_dims == 2:
            names = ("timesteps", "batch") + names
            self.batch_size = data.shape[1]
            self.rollout_length = data.shape[0]
        return names

    # Numpy - PyTorch translations all done here:
    @property
    def numpy(self):
        if not hasattr(self, "_numpy"):
            self._numpy = self._tensor.detach().cpu().numpy()
        return self._numpy
    
    @property
    def tensor(self):
        if not hasattr(self, "_tensor"):
            self._tensor = self._TensorType(self._numpy)
            self._tensor = self._tensor.refine_names(*self._parse_batch_dims(self._tensor))
        return self._tensor
    
    @property
    def _TensorType(self):
        '''How to convert self._numpy to PyTorch tensor (either FloatTensor or LongTensor)'''
        return self.mdp.FloatTensor
        
    @numpy.setter
    def numpy(self, data):
        self._numpy = data
        if hasattr(self, "_tensor"):
            assert not self._tensor.requires_grad
            del self._tensor

    @tensor.setter
    def tensor(self, data):
        full_name = self._parse_batch_dims(data)
        self._tensor = data.refine_names(*full_name)
        if hasattr(self, "_numpy"):
            del self._numpy

    def remove_from_gpu(self):
        if hasattr(self, "_tensor"):
            getattr(self, "numpy")
            assert not self._tensor.requires_grad
            del self._tensor
    
    # these two methods define what are dimensions of tensor and what are names.
    @classmethod
    def rshape(cls):
        '''
        Returns representation shape.
        This list does not consider batch dimensions at the beginning.
        output: torch.Size
        '''
        raise NotImplementedError()
    
    @classmethod
    def rnames(cls):
        '''
        Returns names of dimensions of the representation.
        This list does not consider batch dimensions at the beginning.
        output: tuple of strings
        '''
        raise NotImplementedError()
    
    # interface functions ----------------------------------------------------------------
    @property
    def total_size(self):
        return self.batch_size * self.rollout_length

    def batch(self, indices):
        '''
        Returns subset by given indices
        input: indices - numpy array, ints
        output: Representation
        '''
        data = self.tensor.flatten(["timesteps", "batch"], "batch")
        return type(self)(torch_index(data, indices))

    def append(self, last):
        '''
        Adds one row along "timesteps" axis
        input: last - Representation
        output: Representation
        '''
        if "timesteps" in self.tensor.names:
            return type(self)(torch.cat([self.tensor, last.tensor.align_as(self.tensor)], "timesteps"))
        return type(self)(torch_stack([self.tensor, last.tensor], 0, "timesteps"))

    def detach(self):
        self.tensor = self.tensor.detach()
        return self

    def clamp(self, low, high):
        return type(self)(torch.clamp(self.tensor, low, high))

    def __getitem__(self, idx):
        '''
        TODO
        '''
        if not hasattr(self, "_tensor"):
            if isinstance(idx, Representation):
                return type(self)(self.numpy[idx.numpy])
            else:
                return type(self)(self.numpy[idx])
        elif isinstance(idx, Representation):
            return type(self)(torch_index(self.tensor, idx.tensor))
        elif isinstance(idx, np.ndarray):
            return type(self)(torch_index(self.tensor, idx))      
        return type(self)(self.tensor[idx])

    def __setitem__(self, idx, value):
        '''
        TODO
        '''
        if not hasattr(self, "_tensor"):
            if isinstance(idx, Representation):
                idx = idx.numpy

            if isinstance(value, Representation):
                self.numpy[idx] = value.numpy
            else:
                self.numpy[idx] = value
        elif isinstance(value, Representation):
            self.tensor = torch_set_index(self.tensor, idx, value.tensor)
        else:
            self.tensor = torch_set_index(self.tensor, idx, value)

    def __iadd__(self, other):
        '''
        TODO
        '''
        if not hasattr(self, "_tensor") or isinstance(other, np.ndarray):
            if isinstance(other, Representation):
                self.numpy += other.numpy
            else:
                self.numpy += other
        elif isinstance(other, Representation):
            self.tensor += other.tensor
        else:
            self.tensor += other
        return self

    def __imul__(self, other):
        '''
        TODO
        '''
        if not hasattr(self, "_tensor") or isinstance(other, np.ndarray):
            if isinstance(other, Representation):
                self.numpy *= other.numpy
            else:
                self.numpy *= other
        elif isinstance(other, Representation):
            self.tensor *= other.tensor
        else:
            self.tensor *= other
        return self

    def __mul__(self, other):
        '''
        TODO
        '''
        if not hasattr(self, "_tensor") or isinstance(other, np.ndarray):
            if isinstance(other, Representation):
                return type(self)(self.numpy * other.numpy)
            else:
                return type(self)(self.numpy * other)
        elif isinstance(other, Representation):
            return type(self)(self.tensor * other.tensor.align_as(self.tensor))
        else:
            return type(self)(self.tensor * other)
        return self

    def __add__(self, other):
        '''
        TODO
        '''
        if not hasattr(self, "_tensor") or isinstance(other, np.ndarray):
            if isinstance(other, Representation):
                return type(self)(self.numpy + other.numpy)
            else:
                return type(self)(self.numpy + other)
        elif isinstance(other, Representation):
            return type(self)(self.tensor + other.tensor.align_as(self.tensor))
        else:
            return type(self)(self.tensor + other)
        return self

    def __sub__(self, other):
        '''
        TODO
        '''
        if not hasattr(self, "_tensor") or isinstance(other, np.ndarray):
            if isinstance(other, Representation):
                return type(self)(self.numpy - other.numpy)
            else:
                return type(self)(self.numpy - other)
        elif isinstance(other, Representation):
            return type(self)(self.tensor - other.tensor.align_as(self.tensor))
        else:
            return type(self)(self.tensor - other)
        return self

    __rmul__ = __mul__
    __radd__ = __add__

    @classmethod
    def _default_name(cls):
        '''output: default name of representation, str'''
        return cls.__name__

    def __repr__(self):
        return f"{self._default_name()} {self.numpy.shape}"