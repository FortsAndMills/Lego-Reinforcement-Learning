from LegoRL.representations.representation import Representation

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(Representation):
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
        '''constructs uniform policy'''
        return cls(torch.ones(cls.mdp.num_actions, names=("actions",)).to(cls.mdp.device))

    def __getattr__(self, name):
        return getattr(self.distribution, name)
        
    @classmethod
    def rshape(cls):
        return torch.Size([cls.mdp.num_actions])

    @classmethod
    def rnames(cls):
        return ("actions",)    

    def __repr__(self):    
        return 'Policy for {} actions'.format(self.mdp.num_actions)

