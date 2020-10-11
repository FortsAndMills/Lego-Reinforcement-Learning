from LegoRL.core.RLmodule import RLmodule
from LegoRL.models.model import Model

import os
import torch
import numpy as np

class Trainer(RLmodule):
    '''
    Module for optimization tasks with PyTorch framework.
    
    Args:
        models - list of Models
        optimizer - class of optimizer, torch.optim
        optimizer_args - dict of args for optimizer
        clip_gradients - whether to clip gradients, float or None
    '''
    def _get_nets(self, module):
        '''
        output: list of nn.Module
        '''
        if isinstance(module, Model):
            return [module.net]

        nets = []
        for module in module.modules:
            nets += self._get_nets(module)
        return nets

    def __init__(self, par, models, optimizer=torch.optim.Adam, optimizer_args={}, clip_gradients=None):
        super().__init__(par)
        
        # checkup
        if not isinstance(models, list): models = [models]
        nets = []
        for model in models:
            nets += self._get_nets(model)
        assert len(nets) > 0, "Networks not found"

        # network and optimizer initialization
        self.full_network = torch.nn.ModuleList(nets)
        self.optimizer_args = optimizer_args 
        self.optimizer = optimizer(self.full_network.parameters(), **optimizer_args)   
        self.clip_gradients = clip_gradients 
        
        # precomputing if network has noise for logging purposes
        self._is_noised = False
        for layer in self.full_network.modules():
            if hasattr(layer, "magnitude"):
                self._is_noised = True
    
    def optimize(self, loss):
        # performing optimization step
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_gradients is not None:
            g = torch.nn.utils.clip_grad_norm_(self.full_network.parameters(), self.clip_gradients)
        self.optimizer.step()
        
        # additional logs
        if self.clip_gradients is not None:
            self.log(self.name + " gradient_norm", g.item(), "gradient norm")
        
        if self._is_noised and self.system.time_for_rare_logs():
            self.log(self.name + " magnitude", self.average_magnitude(), "noise magnitude")
    
    # interface functions ----------------------------------------------------------------
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

    def load(self, folder_name):
        path = os.path.join(folder_name, self.name)
        self.optimizer.load_state_dict(torch.load(path))

    def save(self, folder_name):
        path = os.path.join(folder_name, self.name)
        torch.save(self.optimizer.state_dict(), path)

    def hyperparameters(self):
        hp = {}        
        hp["clip_gradients"] = self.clip_gradients
        hp["optimizer"] = type(self.optimizer).__name__
        for name, val in self.optimizer_args.items():
            hp["optimizer " + name] = val
        return hp

    def __repr__(self):
        return "Standard SGD loss optimization"