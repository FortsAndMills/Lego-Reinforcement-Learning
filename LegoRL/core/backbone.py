from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.cache import cached

import torch
import torch.nn as nn
import numpy as np

'''
The scheme for all networks in this framework is as following:
backbone       - main network, provided by user, performing feature extraction, returns (feature_size)
head           - parameterized transformation from (feature_size) to (output_size), typically nn.Linear
representation - interprets (output_size) as policy, Q-function, Categorical Q, etc.

feature_size is provided by user implicitly. 
It is calculated using (input_shape) which is given by modules, providing heads.
Representation class also provides desired shape, i.e. the dimension of output per element in the batch.

Backbone class stores <full_network> field, containing all nn.Modules in nn.ModuleList
for further optimization.
'''

class Backbone(RLmodule, torch.nn.Module):
    '''
    Module for optimization tasks with PyTorch framework
    
    Args:
        network - feature extractor neural net, nn.Module

    Provides: mount_head, full_network, forward
    '''

    def __init__(self, network, *args, **kwargs):
        RLmodule.__init__(self, *args, **kwargs)
        torch.nn.Module.__init__(self)
        
        self.net = network
        self.full_network = nn.ModuleList([self.net])

    def feature_size(self, input_shape):
        '''
        Checks if this backbone takes as input data of input_shape structure
        Then calculates feature_size of backbone output
        input: input_shape - shape of backbone input data, tuple
        output: feature_size - int
        '''
        self.net = self.net.to(self.system.device)
        with torch.no_grad():
            return self.net(torch.randn(2, *input_shape).to(self.system.device)).shape[1]

    def mount_head(self, head):
        '''
        Adds parameters of nn.Module to the list of trainable parameters.
        input: head - nn.Module class
        '''
        self.full_network.append(head)

    @cached
    def forward(self, *input, **kwargs):
        return self.net(*input)

    def has_noise(self):
        '''
        Returns true if this module has at least one Noisy layer.
        output: bool
        '''
        for layer in self.full_network.modules():
            if hasattr(layer, "magnitude"):
                return True
        return False

    def average_magnitude(self):
        '''
        Returns average magnitude of the whole network
        output: float
        '''
        mag, n_params = sum([np.array(layer.magnitude()) for layer in self.full_network.modules() if hasattr(layer, "magnitude")])
        return mag / n_params           
            
    def numel(self):
        '''
        Returns number of trainable parameters in the network
        output: int
        '''
        return sum(p.numel() for p in self.full_network.parameters() if p.requires_grad)

    def _load(self, name):
        self.full_network.load_state_dict(torch.load(name + "-" + self.name))

    def _save(self, name):
        torch.save(self.full_network.state_dict(), name + "-" + self.name)    

    def __repr__(self):
        return f"Backbone of network"