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
        # PyTorch NamedTensor issues again...
        mu = F.tanh(self.tensor.align_to("musigma", ...)[0].rename(None))
        sigma = F.softplus(self.tensor.align_to("musigma", ...)[1].rename(None))
        
        return Normal(mu, sigma)

    # TODO
    def log_prob(self, actions):
        component_prob = self.distribution.log_prob(actions)
        return component_prob.sum(-1)

    # TODO
    def entropy(self):
        component_entr = self.distribution.entropy()
        return component_entr.sum(-1)

    def __getattr__(self, name):
        return getattr(self.distribution, name)
    
    @classmethod
    def rshape(cls):
        return torch.Size([2, *cls.mdp.action_shape])

    @classmethod
    def rnames(cls):
        return ("musigma",) + tuple("actionsI" + "I"*k for k in range(len(cls.mdp.action_shape)))

    @classmethod
    def _default_name(cls):   
        return 'Gaussian Policy'

