from LegoRL.core.RLmodule import RLmodule
from LegoRL.core.composed import Reference, ReferenceList

import torch

class Trainer(RLmodule):
    '''
    Module for optimization tasks with PyTorch framework.
    Conceptually, it trains one backbone network with several heads.
    
    Args:
        backbone - RLmodule with "full_network" property
        losses - list of losses to optimize, list of modules with "loss" method
        weights - list of coeffecients for these losses
        optimizer - class of optimizer, torch.optim
        optimizer_args - dict of args for optimizer
        clip_gradients - whether to clip gradients, float or None
    '''
    def __init__(self, backbone, losses=[], weights=None, optimizer=torch.optim.Adam, optimizer_args={}, clip_gradients=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.backbone = Reference(backbone)
        self.losses = ReferenceList(losses)
        self.weights = weights or [1.] * len(self.losses)
        assert len(self.weights) == len(self.losses), "Error: there must be weights per each loss"
        assert len(self.losses) > 0, "Error: there must be at least one loss function"
            
        self._optimizer_initialized = False  
        self._is_noised = None      
        self.network = None
        self.clip_gradients = clip_gradients 
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

    def _init_optimizer(self):
        '''
        Optimizer initialization happens on the first training step
        to avoid initialization before the last head connects to backbone
        '''
        if not self._optimizer_initialized:
            self._optimizer_initialized = True
            self._is_noised = self.backbone.has_noise()
            self.network = self.backbone.full_network
            self.optimizer = self.optimizer(self.network.parameters(), **self.optimizer_args)
    
    def iteration(self):
        '''
        Makes one step of training using sum of losses.
        '''
        assert len(self.losses) > 0, "Error: network with no losses is attempted to be optimized"
        self._init_optimizer()        
        
        # Loss collection. If some loss is None, then there is no data yet
        # and no optimization happens.
        self.debug("initiates loss computation.", open=True)
        full_loss = 0
        for weight, loss_provider in zip(self.weights, self.losses):
            loss = loss_provider.loss()
            if loss is None:
                self.debug("loss is None; no optimization step is performed.", close=True)
                return

            full_loss += weight * loss

            self.log(loss_provider.name, weight * loss.detach().cpu().numpy(), "training iteration", "loss")

        # performing optimization step and logging.
        self.debug("performs optimization step.", close=True)
        self.optimizer.zero_grad()
        full_loss.backward()
        if self.clip_gradients is not None:
            self.system.log(self.name + " gradient_norm", torch.nn.clip_grad_norm_(self.network.parameters(), self.clip_gradients), "training iteration", "gradient norm")
        self.optimizer.step()
        
        if len(self.losses) > 1:
            self.system.log(self.backbone.name + " loss", full_loss.detach().cpu().numpy(), "training iteration", "loss")
        
        if self._is_noised and self.system.time_for_rare_logs():
            self.log(self.backbone.name + " magnitude", self.backbone.average_magnitude(), "training iteration", "noise magnitude")
    
    def _load(self, name):
        self._init_optimizer()
        self.optimizer.load_state_dict(torch.load(name + "-" + self.name))

    def _save(self, name):
        torch.save(self.optimizer.state_dict(), name + "-" + self.name)

    def __repr__(self):
        return f"Trains <{self.backbone.name}> with all heads using following losses: " + ", ".join(["<" + loss_provider.name + ">" for loss_provider in self.losses])