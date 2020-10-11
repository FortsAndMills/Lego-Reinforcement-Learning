from LegoRL.losses.loss import Loss

from LegoRL.utils.namedTensorsUtils import torch_max

class ProximalCriticLoss(Loss):
    """
    PPO loss for critic.
    """
    def __init__(self, par, cliprange=0.2, *args, **kwargs):
        super().__init__(par, *args, **kwargs)
        self.cliprange = cliprange
    
    def batch_loss(self, V, old_V, target, *args, **kwargs):
        '''
        Calculates loss for batch based on TD-error from DQN algorithm.
        input: V
        input: old_V - V
        input: target - V
        output: Loss
        '''
        old_V = old_V.detach()
        V2 = old_V + (V - old_V).clamp(-self.cliprange, self.cliprange)
        
        loss1 = V.compare(target.detach())
        loss2 = V2.compare(target.detach())

        # NamedTensors Issue
        return self.mdp["Loss"](torch_max(loss1.tensor, loss2.tensor))

    def hyperparameters(self):
        return {"cliprange": self.cliprange}
        
    def __repr__(self):
        return f"Calculates proximal TD loss by clipping prediction when it is far from old prediction"