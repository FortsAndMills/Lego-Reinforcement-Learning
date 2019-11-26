from LegoRL.representations.representation import Representation

import torch
from torch.distributions import Categorical

class Policy(Representation):
    '''
    FloatTensor representing policy, (*batch_shape x num_actions)
    '''
    @classmethod
    def shape(cls, system):
        return torch.Size([system.num_actions])

    @classmethod
    def names(cls):
        return ("actions",)

    def __getattr__(self, name):
        return getattr(Categorical(logits=self.tensor.rename(None)), name)

    def __repr__(self):    
        return 'Policy for {} actions'.format(self.system.num_actions)

