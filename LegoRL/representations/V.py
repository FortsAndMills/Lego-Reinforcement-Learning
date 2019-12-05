from LegoRL.representations.representation import Representation

import torch

class V(Representation):
    """
    Value function representation.
    """
    @classmethod
    def shape(cls, system):
        return torch.Size([])
    
    @classmethod
    def names(cls):
        return tuple()
    
    @classmethod
    def constructor(cls):
        '''
        Returns class fabric, responsible for each dimension of the tensor.
        output: dict,
            keys - dimensions names, str
            values - class fabrics
        '''
        return {}

    def construct(self, tensor):
        '''
        Returns same class as cls with fabrics 
        corresponding to names of dimensions of tensor.
        input: tensor - FloatTensor
        output: V with dimensions as in input tensor
        '''
        result = V
        for name, fabric in self.constructor().items():
            if name in tensor.names:
                result = fabric(result)
        return result(self.system, tensor)
        
    def one_step(self, batch):
        '''
        Calculates one-step approximation using this tensor as V(s') estimation
        input: Batch
        output: V (dimensions not changed)
        '''
        rewards = batch.rewards.align_as(self.tensor)
        discounts = batch.discounts.align_as(self.tensor)
        return self.construct(rewards + discounts * self.tensor)

    def compare(self, target):
        '''
        Calculates loss using model prediction and given target ("guess")
        input: V
        output: FloatTensor, (*batch_shape)
        '''
        return (target.tensor - self.tensor).pow(2)    

    def subtract_v(self, v):
        '''
        Estimates advantage by reducing v.
        input: V
        output: V (dimensions not changed)
        '''
        return self.construct(self.tensor - v.tensor.align_as(self.tensor))

    def add_v(self, v):
        '''
        Turns advantage to Q by adding v.
        input: V
        output: V (dimensions not changed)
        '''
        return self.construct(self.tensor + v.tensor.align_as(self.tensor))

    def scalar(self):        
        '''
        Returns scalar version of this representation without additional dimensions.
        output: FloatTensor, (*batch_shape)
        '''
        return self.tensor

    def value(self):
        return self

    def __repr__(self):    
        return 'V-function'