from LegoRL.representations.policy import Policy

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class GaussianPolicy(Policy):
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
        mu = torch.tanh(self.tensor.align_to("musigma", ...)[0].rename(None))
        sigma = F.softplus(self.tensor.align_to("musigma", ...)[1].rename(None))
        
        return MultivariateNormal(mu, torch.diag_embed(sigma))

    # def log_prob(self, actions):
    #     '''
    #     input: Action
    #     output: FloatTensor
    #     '''
    #     #NamedTensors issue
    #     component_prob = self.distribution.log_prob(actions.tensor.rename(None))
    #     return component_prob

    # def entropy(self):
    #     '''
    #     output: FloatTensor
    #     '''
    #     component_entr = self.distribution.entropy()
    #     return component_entr
    
    @classmethod
    def rshape(cls):
        return torch.Size([2, *cls.mdp.action_shape])

    @classmethod
    def rnames(cls):
        return ("musigma",) + tuple("actionsI" + "I"*k for k in range(len(cls.mdp.action_shape)))

    @classmethod
    def _default_name(cls):   
        return 'Gaussian Policy'

