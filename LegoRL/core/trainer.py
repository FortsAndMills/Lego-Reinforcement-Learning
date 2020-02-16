from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.reference import Reference, ReferenceList

import os
import torch
import numpy as np

class Trainer(RLmodule):
    '''
    Module for optimization tasks with PyTorch framework.
    
    Args:
        transformations - list of modules with "net" property
        losses - list of losses to optimize, list of modules with "loss" method
        weights - list of coeffecients for these losses
        optimizer - class of optimizer, torch.optim
        optimizer_args - dict of args for optimizer
        clip_gradients - whether to clip gradients, float or None
    '''
    def __init__(self, transformations, losses=[], weights=None, optimizer=torch.optim.Adam, optimizer_args={}, clip_gradients=None, frozen=False, timer=1):
        super().__init__(frozen=frozen, timer=timer)
        
        self.transformations = ReferenceList(transformations)
        self.losses = ReferenceList(losses)
        self.weights = weights or [1.] * len(self.losses)
        
        assert len(self.weights) == len(self.losses), "Error: there must be weights per each loss"
        assert len(self.losses) > 0, "Error: there must be at least one loss function"
                
        self.clip_gradients = clip_gradients 
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

    def _initialize(self):
        # guarantee all networks are initialized
        for tf in self.transformations:
            tf.initialize()
        
        # network and optimizer initialization
        self.full_network = torch.nn.ModuleList([tf.net for tf in self.transformations])  
        self.optimizer = self.optimizer(self.full_network.parameters(), **self.optimizer_args)      
        
        # precomputing if network has noise for logging purposes
        self._is_noised = False
        for layer in self.full_network.modules():
            if hasattr(layer, "magnitude"):
                self._is_noised = True
    
    def _iteration(self):
        # Loss collection. 
        # If some loss is None, then there is no data yet, and no optimization happens.
        self.debug("initiates loss computation.", open=True)

        full_loss = 0
        for weight, loss_provider in zip(self.weights, self.losses):
            loss = loss_provider.loss()
            if loss is None:
                self.debug("loss is None; no optimization step is performed.", close=True)
                return

            full_loss += weight * loss

            self.log(loss_provider.name, (weight * loss).detach().cpu().numpy(), f"{self.name} loss")

        # performing optimization step
        self.debug("performs optimization step.", close=True)
        self.optimizer.zero_grad()
        full_loss.backward()
        if self.clip_gradients is not None:
            g = torch.nn.utils.clip_grad_norm_(self.full_network.parameters(), self.clip_gradients)
        self.optimizer.step()
        
        # additional logs
        if self.clip_gradients is not None:
            self.log(self.name + " gradient_norm", g, "gradient norm")

        if len(self.losses) > 1:
            self.log(self.name + " full loss", full_loss.detach().cpu().numpy(), f"{self.name} loss")
        
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

    def _load(self, folder_name):
        path = os.path.join(folder_name, self.name)
        self.optimizer.load_state_dict(torch.load(path))

    def _save(self, folder_name):
        path = os.path.join(folder_name, self.name)
        torch.save(self.optimizer.state_dict(), path)

    def __repr__(self):
        return (f"Trains " + 
                ", ".join(["<" + transformation.name + ">" for transformation in self.transformations]) + 
                " using following losses: " + 
                ", ".join(["<" + loss_provider.name + ">" for loss_provider in self.losses]))