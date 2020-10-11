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

    def raw_embedding(self):
        return self.mdp[Embedding(self.rshape().numel())](self.tensor.rename(...).flatten(list(self.rnames()), "features"))
    
    #def compare(self, other):
    #    return self.raw_embedding().compare(other.raw_embedding())

    @classmethod
    def _default_name(cls):   
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

    def raw_embedding(self):
        preprocess = self.mdp.action_preprocessing(self.tensor)
        return self.mdp[Embedding(self.rshape().numel())](preprocess.rename(...).flatten(list(self.rnames()), "features"))

    @property
    def _TensorType(self):
        return self.mdp.ActionTensor

    @classmethod
    def _default_name(cls):   
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

    @classmethod
    def _default_name(cls):   
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

    @classmethod
    def _default_name(cls):   
        return "Discount Representation"

class Flag(Representation):
    '''
    Representation class for flags.
    '''    
    @classmethod
    def rshape(cls):
        return torch.Size([])

    @classmethod
    def rnames(cls):
        return tuple()

    @property
    def _TensorType(self):
        return self.mdp.BoolTensor

    @classmethod
    def _default_name(cls):   
        return "Flag Representation"

def Embedding(size=0, emb_name="Embedding"):
    '''
    Embedding representation (arbitrary shaped vector)
    Args:
        size - int, scalar if zero
    '''
    class Embedding(Representation):
        embedding_size = size

        @classmethod
        def rshape(cls):
            return torch.Size([cls.embedding_size]) if cls.embedding_size else torch.Size([])

        @classmethod
        def rnames(cls):
            return ("features",) if cls.embedding_size else tuple()

        def compare(self, other):
            cmp = (self.tensor - other.tensor)**2
            return self.mdp["Loss"](cmp.sum(dim="features") if self.embedding_size else cmp)

        # TODO: error, for loss it is not working
        @classmethod
        def _default_name(cls):
            return emb_name + f" of size {cls.embedding_size}" if cls.embedding_size else ""
    
    return Embedding