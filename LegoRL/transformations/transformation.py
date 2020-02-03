from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.cache import cached_forward
from LegoRL.buffers.storage import Which

import torch

'''
Transformation is RLmodule for neural networks.
Its input and output must be Representation (i.e. State, Embedding, Q-function, etc.)
The network itself must take input_size and output_size as constructor arguments!
'''

class Transformation(RLmodule):
    '''
    Module for optimization tasks with PyTorch framework
    
    Args:
        network - nn.Module, taking two constructor parameters:
            input_size (int)
            output_size (int)

    Provides:
        __call__ - function, performing transformation for given storage
        output_representation - property, returning Representation class for output
    '''
    def __init__(self, network=None):
        super().__init__()        
        self.net = network

    def _initialize(self):
        print(f"Initializing <{self.name}>:")

        input_shape = self.input_representation.rshape()
        print(f"  Input shape is {input_shape}")
        output_shape = self.output_representation.rshape()
        print(f"  Output shape is {output_shape}")   

        self.net = self.net(input_shape.numel(), output_shape.numel()).to(self.mdp.device)

    @property
    def input_representation(self):
        '''output: Representation class'''
        raise NotImplementedError()

    @property
    def output_representation(self):
        '''output: Representation class'''
        raise NotImplementedError()

    def _get_input(self, storage, which):
        '''
        input: Storage
        input: which - Which marker (which states to get from rollout of states)
        output: input for network
        '''
        raise NotImplementedError()

    @cached_forward
    def __call__(self, storage, which=Which.current):
        input = self._get_input(storage, which)
        output = self.net(input)
        return self.output_representation.from_linear(output)

    def _load(self, name):
        self.net.load_state_dict(torch.load(name + "-" + self.name))

    def _save(self, name):
        torch.save(self.net.state_dict(), name + "-" + self.name)    

    def __repr__(self):
        return f"Undefined transformation"