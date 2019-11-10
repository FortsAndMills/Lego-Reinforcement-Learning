from .RLmodule import *

class Trainer(RLmodule):
    '''
    Module for optimization tasks with PyTorch framework.
    Conceptually, it trains one backbone network with several heads.
    
    Args:
        backbone - RLmodule with "full_network" property
        losses - list of losses to optimize, list of modules with "loss" method
        optimizer - class of optimizer, torch.optim
        optimizer_args - dict of args for optimizer
        clip_gradients - whether to clip gradients, float or None
    '''
    def __init__(self, backbone, losses=[], optimizer=optim.Adam, optimizer_args={}, clip_gradients=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.backbone = Reference(backbone)
        self.losses = ReferenceList(losses) 
                    
        self.network = None
        self.optimizer_initialized = False
        self.clip_gradients = clip_gradients 
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

    def init_optimizer(self):
        '''
        Optimizer initialization happens on the first training step
        to avoid initialization before the last head connects to backbone
        '''
        if not self.optimizer_initialized:
            self.network = self.backbone.full_network
            self.optimizer_initialized = True
            self.optimizer = self.optimizer(self.network.parameters(), **self.optimizer_args)
    
    def iteration(self):
        '''
        Makes one step of training using sum of losses.
        '''
        assert len(self.losses) > 0, "Error: network with no losses is attempted to be optimized"
        self.init_optimizer()
        
        # Loss collection. If some loss is None, then there is no data yet
        # and no optimization happens.
        self.debug("starts loss calculation", +1)
        full_loss = 0
        for loss_provider in self.losses:
            loss = loss_provider.loss()            
            if loss is None:
                self.debug("Loss is None; no optimization step is performed", -1)
                return

            full_loss += loss

        self.debug("performs optimization step.", -1)
        self.optimizer.zero_grad()
        full_loss.backward()
        if self.clip_gradients is not None:
            self.system.log(self.name + " gradient_norm", clip_grad_norm_(self.network.parameters(), self.clip_gradients), "training iteration", "gradient norm")
        self.optimizer.step()
        
        if len(self.network) > 2:
            self.system.log(self.name + " loss", full_loss.detach().cpu().numpy(), "training iteration", "loss")
        
        # TODO: noisy layers
        #self.log(self.name + "_magnitude", self.average_magnitude(), "magnitude logging iteration", "noise magnitude")
    
    def load(self, name):
        self.init_optimizer()
        self.optimizer.load_state_dict(torch.load(name + "-" + self.name))

    def save(self, name):
        torch.save(self.optimizer.state_dict(), name + "-" + self.name)

    def __repr__(self):
        loss_names = [loss_provider.name for loss_provider in self.losses]
        return f"Trains {self.backbone.name} with all heads using following losses: " + ",".join(loss_names)