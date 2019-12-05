import torch
import numpy as np

def Quantile(num_atoms=51):
    """
    Quantile value functions.
    Adds new dimension to representation tensor with num_atoms elements.
    Output interpreted as uniform distributions with num_atoms outcomes.
    Based on: https://arxiv.org/abs/1710.10044
    
    Args:
        num_atoms - number of atoms in approximation distribution, int
    """

    def Quantile(parclass):
        class QuantileValue(parclass):
            @classmethod
            def shape(cls, system):
                return torch.Size((num_atoms,)) + super().shape(system)
        
            @classmethod
            def names(cls):
                return ("atoms",) + super().names()

            @classmethod
            def constructor(cls):
                dims = super().constructor()
                dims["atoms"] = Quantile
                return dims

            def _expectation(self):
                '''
                Reduces atoms dimension.
                output: V without atoms dimension
                '''
                return self.construct(self.tensor.sum(dim="atoms") / num_atoms)

            def compare(self, target):
                '''
                Calculates Wasserstein distance between target and this Quantile value.
                input: target - QuantileV
                output: FloatTensor, (*batch_shape)
                '''
                target = target.tensor.unflatten('atoms', [('atoms', 1),         ('atomsII', num_atoms)])
                q      =   self.tensor.unflatten('atoms', [('atoms', num_atoms), ('atomsII', 1)])
                diff = target - q

                tau = self.system.FloatTensor((2 * np.arange(num_atoms) + 1) / (2.0 * num_atoms), names=("atoms",)).align_as(diff)
                
                # mean should be taken across "atomsII" dimension.
                # NamedTensors mean issue...
                return (diff * (tau - (diff < 0).float())).sum('atomsII').sum('atoms') / num_atoms

            def greedy(self):
                return self._expectation().greedy()
                
            def value(self, policy=None):
                if "actions" not in self.names():
                    return self
                if policy is None:
                    return self.gather(self.greedy())
                return super().value(policy)

            def scalar(self):
                return self._expectation().scalar()

            def __repr__(self):    
                return super().__repr__() + f' in quantile form with {num_atoms} atoms'
        return QuantileValue
    return Quantile