from .RLmodule import *
from torch.nn.utils import clip_grad_norm_

'''
The scheme for all networks in this framework is as following:
backbone - main network, provided by user, performing feature extraction, returns (feature_size)
head     - parameterized transformation from (feature_size) to (output_size), typically nn.Linear
hat      - transforms (output_size) to desired representation (policy, Q-function, Categorical Q, etc.)

feature_size is provided by user implicitly. 
It is calculated using (input_shape) which can be given by modules, providing heads.
Hat class provides (output_size), i.e. the dimension of output per element in the batch.

Backbone class stores <full_network> field, containing all nn.Modules in nn.ModuleList,
and provides each head with <net> field, containing (backbone, head, hat) in nn.Sequential.

This nn.Sequential is wrapped into NetworkWithCache, which stores output for batch
inside the batch and reuses the cache if the network is called again for the same batch.

In this modular framework cache is crucial as different modules may call the same heads
for the same batch without knowing about each other.
Consider Twin DQN with shared replay buffer as an example why the cache is needed.
'''

class Hat(nn.Module):
    def required_shape(self):
        '''
        Returns desired shape of output per each object in the batch
        output: int
        '''
        raise NotImplementedError

    def extra_repr(self):
        '''
        Returns string description of this module for PyTorch nn.Module representation
        output: str
        '''
        raise NotImplementedError

class NetworkWithCache(nn.Module):
    '''
    Constructs a sequential (backbone, head, hat) with cache
    Args: 
        backbone_name - name of backbone RLmodule, str
        head_name - name of head RLmodule, str
        backbone, head, hat - nn.Modules
    '''
    def __init__(self, system, backbone_name, head_name, backbone, head, hat):
        super().__init__()
        self.system = system
        self.head_name = head_name
        self.backbone_name = backbone_name

        self.backbone = backbone
        self.head = nn.Sequential(head, hat)
    
    def forward(self, *input, **kwargs):
        # if storage is not provided, the cache is considered to be empty.
        storage = kwargs.get("storage", {})
        if "storage" in kwargs: del kwargs["storage"]

        # if head output is cached, we just return it.
        if self.head_name in storage:
            self.system.debug(self.head_name, "head output is reused from cache!")
            return storage[self.head_name]
        
        # if backbone output is cached, we reuse it
        # and store head output in cache
        if self.backbone_name in storage:
            self.system.debug(self.backbone_name, "backbone output is reused from cache!")
            backbone_output = storage[self.backbone_name]
            
            output = self.head(backbone_output)
            storage[self.head_name] = output
            return output

        # if nothing is cached, we make forward pass
        # and cache backbone output (for other heads)
        # and cache head output
        backbone_output = self.backbone(*input, **kwargs)
        storage[self.backbone_name] = backbone_output
        
        output = self.head(backbone_output)
        storage[self.head_name] = output
        return output

class Backbone(RLmodule):
    '''
    Module for optimization tasks with PyTorch framework
    
    Args:
        backbone - feature extractor neural net, nn.Module

    Provides: mount_head, full_network
    '''

    def __init__(self, backbone, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_network = nn.ModuleList([backbone.to(device)])

    @property
    def backbone(self):
        return self.full_network[0]

    def mount_head(self, head_name, input_shape, head_network, hat):
        '''
        Checks if this backbone takes as input data of input_shape structure
        Then calculates feature_size of backbone output
        Constructs a head using head_network class with head as top
        input: head_name - name of head RLmodule, str
        input: input_shape - shape of backbone input data, tuple
        input: head_network - nn.Module class
        input: hat - nn.Module, inherited from Hat
        output: _NetworkWithCache, consisting of backbone, head_network and head
        '''
        with torch.no_grad():
            feature_size = self.backbone(Tensor(2, *input_shape)).shape[1]

        print(f"Adding new head {head_name} to {self.name}:")
        print(f"  Input shape is {input_shape}")
        print(f"  Backbone feature size is {feature_size}")
        print(f"  Desired output is {hat.required_shape()}")            
        
        head = head_network(feature_size, hat.required_shape()).to(device)
        self.full_network.append(head)
        return NetworkWithCache(self.system, self.name, head_name, self.backbone, head, hat)

    def average_magnitude(self):
        '''
        Returns average magnitude of the whole network
        output: float
        '''
        mag, n_params = sum([np.array(layer.magnitude()) for layer in self.full_network.modules() if hasattr(layer, "magnitude")])
        return mag / n_params           
            
    def numel(self):
        '''
        Returns number of parameters in the network
        output: int
        '''
        return sum(p.numel() for p in self.full_network.parameters() if p.requires_grad)

    def load(self, name):
        self.full_network.load_state_dict(torch.load(name + "-" + self.name))

    def save(self, name):
        torch.save(self.full_network.state_dict(), name + "-" + self.name)    

    def __repr__(self):
        return f"Backbone of network"