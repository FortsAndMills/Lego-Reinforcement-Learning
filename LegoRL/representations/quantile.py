import torch
import numpy as np

def Quantile(parclass, num_atoms=51):
    """
    Quantile value functions.
    Adds new dimension to representation tensor with num_atoms elements.
    Output interpreted as uniform distributions with num_atoms outcomes.
    Based on: https://arxiv.org/abs/1710.10044
    
    Args:
        num_atoms - number of atoms in approximation distribution, int
    """
    tau = torch.tensor((2 * np.arange(num_atoms) + 1) / (2.0 * num_atoms), names=("atoms",))

    class QuantileValue(parclass):
        def expectation(self):
            '''
            Reduces atoms dimension by computing expectation.
            output: V (atoms dimension reduced)
            '''
            return self.construct(self.tensor.sum(dim="atoms") / num_atoms)

        def compare(self, target):
            '''
            Calculates Wasserstein distance between target and this Quantile value.
            input: target - V, same dimensions as this
            output: Loss
            '''
            target = target.tensor.unflatten('atoms', [('atoms', 1),         ('atomsII', num_atoms)])
            q      =   self.tensor.unflatten('atoms', [('atoms', num_atoms), ('atomsII', 1)])
            
            diff = target - q 
            cmp = diff * (tau.to(diff.device).align_as(diff) - (diff < 0).float())        
            
            # mean should be taken across "atomsII" dimension.
            # NamedTensors mean issue...
            return self.mdp["Loss"](cmp.sum('atomsII').sum('atoms') / num_atoms)

        def greedy(self):
            return self.expectation().greedy()
            
        def value(self, policy=None):
            if policy is None:
                return self.gather(self.greedy())
            return super().value(policy)

        def scalar(self):
            return self.expectation().scalar()

        @classmethod
        def rshape(cls):
            return torch.Size((num_atoms,)) + super().rshape()
    
        @classmethod
        def rnames(cls):
            return ("atoms",) + super().rnames()

        @classmethod
        def constructor(cls):
            dims = super().constructor()
            dims["atoms"] = lambda parclass: Quantile(parclass, num_atoms)
            return dims

        def __repr__(self):    
            return super().__repr__() + f' in quantile form with {num_atoms} atoms'
    return QuantileValue