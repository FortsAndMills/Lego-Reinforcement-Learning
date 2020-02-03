from LegoRL.representations.representation import Representation

import torch

class State(Representation):
    '''
    Representation class for states.
    '''    
    @classmethod
    def rshape(cls):
        return torch.Size(cls.mdp.observation_shape)

    @classmethod
    def rnames(cls):
        return tuple("observationI" + "I"*k for k in range(len(cls.mdp.observation_shape)))

    def __repr__(self):
        return "State Representation"

class Action(Representation):
    '''
    Representation class for actions.
    '''    
    @classmethod
    def rshape(cls):
        return torch.Size(cls.mdp.action_shape)

    @classmethod
    def rnames(cls):
        return tuple("actionI" + "I"*k for k in range(len(cls.mdp.action_shape)))

    @property
    def _TensorType(self):
        return self.mdp.ActionTensor

    def __repr__(self):
        return "Action Representation"

class Reward(Representation):
    '''
    Representation class for rewards.
    '''    
    @classmethod
    def rshape(cls):
        return torch.Size([])

    @classmethod
    def rnames(cls):
        return tuple()

    def __repr__(self):
        return "Reward Representation"

class Discount(Representation):
    '''
    Representation class for discounts.
    '''    
    @classmethod
    def rshape(cls):
        return torch.Size([])

    @classmethod
    def rnames(cls):
        return tuple()

    def __repr__(self):
        return "Discount Representation"

def Embedding(embedding_size=0, emb_name="Embedding"):
    '''
    Embedding representation (arbitrary shaped vector)
    Args:
        embedding_size - int, scalar if zero
    '''
    class Embedding(Representation):
        @classmethod
        def rshape(cls):
            return torch.Size([embedding_size]) if embedding_size else torch.Size([])

        @classmethod
        def rnames(cls):
            return ("features",) if embedding_size else tuple()

        def compare(self, other):
            cmp = (self.tensor - other.tensor)**2
            return cmp.sum(dim="features") if embedding_size else cmp

        def __repr__(self):
            return emb_name + f" of shape {embedding_size}" if embedding_size else ""
    return Embedding