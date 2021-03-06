from LegoRL.representations.policy import Policy

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class DiscretePolicy(Policy):
    '''
    FloatTensor representing policy, (*batch_shape x num_actions)
    '''
    @property
    def distribution(self):
        '''
        Returns torch.Distribution policy
        output: torch.Categorical
        '''        
        # PyTorch NamedTensor issues again...
        return Categorical(logits=self.tensor.rename(None))

    @property
    def proba(self):
        '''
        Returns torch.Distribution policy
        output: Tensor, (*batch_shape, actions)
        '''        
        return F.softmax(self.tensor, "actions")

    @classmethod
    def uniform(cls):
        '''
        Constructs uniform policy
        output: Policy
        '''
        return cls(torch.ones(cls.mdp.num_actions, names=("actions",)).to(cls.mdp.device))

    @classmethod
    def rshape(cls):
        return torch.Size([cls.mdp.num_actions])

    @classmethod
    def rnames(cls):
        return ("actions",)    

    @classmethod
    def _default_name(cls): 
        return 'Discrete Policy'

