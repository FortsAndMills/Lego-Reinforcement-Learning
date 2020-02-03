from LegoRL.representations.representation import Representation

import torch
import torch.nn.functional as F
from torch.distributions import Normal

class GaussianPolicy(Representation):
    '''
    FloatTensor representing gaussian policy, (*batch_shape x 2 x *action_shape)
    '''
    @property
    def distribution(self):
        '''
        Returns torch.Distribution policy
        output: torch.Normal
        '''
        mu = self.tensor.get(index=0, dim="musigma")
        sigma = F.softplus(self.tensor.get(index=1, dim="musigma"))
        
        # PyTorch NamedTensor issues again...
        return Normal(mu.rename(None), sigma.rename(None))

    def __getattr__(self, name):
        return getattr(self.distribution, name)
    
    @classmethod
    def rshape(cls):
        return torch.Size([2, *cls.mdp.action_shape])

    @classmethod
    def rnames(cls):
        return ("musigma",) + ("actionsI" + "I"*k for k in range(len(cls.mdp.action_shape)))

    def __repr__(self):    
        return 'Gaussian policy'

