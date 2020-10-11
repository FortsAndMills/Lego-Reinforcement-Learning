from LegoRL.representations.policy import Policy
from LegoRL.representations.standard import Action

import torch

class DeterministicPolicy(Policy):
    '''
    FloatTensor representing gaussian policy, (*batch_shape x 2 x *action_shape)
    '''
    @property
    def distribution(self):
        '''
        Returns torch.Distribution policy
        output: Tensor
        '''        
        return torch.tanh(self.tensor)

    def sample(self):
        '''
        output: Action
        '''
        return self.mdp[Action](self.distribution)

    def rsample(self):
        '''
        output: Action
        '''
        return self.mdp[Action](self.distribution)

    def log_prob(self, actions):
        raise Exception("Deterministic Policy does not have log probabilities")

    def entropy(self):
        raise Exception("Deterministic Policy does not have entropy")
    
    @classmethod
    def rshape(cls):
        return torch.Size(cls.mdp.action_shape)

    @classmethod
    def rnames(cls):
        return tuple("actionI" + "I"*k for k in range(len(cls.mdp.action_shape)))

    @classmethod
    def _default_name(cls):   
        return 'Deterministic Policy'

