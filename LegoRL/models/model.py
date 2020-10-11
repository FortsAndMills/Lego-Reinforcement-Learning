#from LegoRL.representations.representation import Which
from LegoRL.core.RLmodule import RLmodule
from LegoRL.representations.standard import State, Embedding
#from LegoRL.core.cache import cached_forward

import os
import torch

'''
Model is RLmodule for neural networks.
Its input and output must be Representation (i.e. State, Embedding, Q-function, etc.)
The network itself must take input_size and output_size as constructor arguments!
'''

class Model(RLmodule):
    '''
    Module for optimization tasks with PyTorch framework
    
    Args:
        network - nn.Module, taking two constructor parameters:
            input_size (int)
            output_size (int)
        input - Representation or list of Representation
        output - Representation
        unite_inputs - bool, if true, all inputs will be concatenated

    Provides:
        __call__ - function, performing model for given storage
    '''
    def __init__(self, par, network=torch.nn.Linear, input=State, output=None, unite_inputs=True):
        super().__init__(par)
        assert output is not None
        
        self.net = network
        self._unite_inputs = unite_inputs
        
        if isinstance(input, list):
            self.input_representation = [self.mdp[inp] for inp in input]
        else:
            self.input_representation = self.mdp[input]
        self.output_representation = self.mdp[output]

        if self.net:
            print(f"Initializing <{type(self).__name__}>:")

            input_numel = 0
            if isinstance(input, list):
                print(f"  Input shapes:")
                for inp in self.input_representation:
                    input_shape = inp.rshape()
                    print(f"    {input_shape}")
                    input_numel += input_shape.numel()
            else:
                input_shape = self.input_representation.rshape()
                print(f"  Input shape is {input_shape}")
                input_numel = input_shape.numel()

            output_shape = self.output_representation.rshape()
            print(f"  Output shape is {output_shape}")   

            self.net = self.net(input_numel, output_shape.numel()).to(self.mdp.device)

    def __call__(self, *input, memory=None):
        '''
        input: arguments for model, Representation
        output: Representation
        '''
        if len(input) > 1 and self._unite_inputs:
                input = tuple(data.raw_embedding().tensor for data in input)
                input = (torch.cat(input, dim="features"),)
        else:
            input = tuple(data.tensor for data in input)

        if memory is None:
            output = self.net(*input)
        else:
            output, memory = self.net(*input, memory)
        
        output = self.output_representation.from_linear(output)

        if memory is None:
            return output
        else:
            return output, memory

    def load(self, folder_name):
        path = os.path.join(folder_name, self.name)            
        self.net.load_state_dict(torch.load(path))

    def save(self, folder_name):
        path = os.path.join(folder_name, self.name)
        torch.save(self.net.state_dict(), path)    

    def __repr__(self):
        return f"Models {self.output_representation._default_name()}"