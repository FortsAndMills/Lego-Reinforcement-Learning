from LegoRL.representations.representation import Representation

import torch

class V(Representation):
    """
    Value function representation.

    A chain of subclasses can inherit from V to create more complex representations.
    Subclasses basically add another dimension to representation, i.e. Q would add "actions" dim.
    """        
    def one_step(self, rewards, discounts):
        '''
        Calculates one-step approximation using this tensor as V(s') estimation
        input: Reward
        input: Discount
        output: V (dimensions not changed)
        '''
        rewards = rewards.tensor.align_as(self.tensor)
        discounts = discounts.tensor.align_as(self.tensor)
        return self.construct(rewards + self.tensor * discounts)

    def compare(self, target):
        '''
        Calculates loss using model prediction and given target ("guess")
        input: target - V (with same dimensions)
        output: Loss
        '''
        return self.mdp["Loss"]((target.tensor - self.tensor).pow(2))  

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
        Returns scalar value for each element in batch
        output: V (all extra dimensions reduced)
        '''
        return self

    @classmethod
    def rshape(cls):
        return torch.Size([])
    
    @classmethod
    def rnames(cls):
        return tuple()
    
    @classmethod
    def constructor(cls):
        '''
        Returns class fabric, responsible for each dimension of the tensor.
        When some dimension is reduced, this constructor is used to create emerged class.
        output: dict,
            keys - dimensions names, str
            values - class fabrics
        '''
        return {}

    def construct(self, tensor):
        '''
        Returns class corresponding to names of dimensions of given tensor.
        input: tensor - FloatTensor
        output: V with dimensions as in input tensor
        '''
        result = V
        for name, fabric in self.constructor().items():
            if name in tensor.names:
                result = fabric(result)
        ans = self.mdp[result](tensor)
        return ans
    
    @classmethod
    def _default_name(cls):   
        return 'V-function'