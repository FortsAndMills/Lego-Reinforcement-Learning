from LegoRL.representations.V import V

import torch
import torch.nn.functional as F

def Categorical(parclass, Vmin=-10, Vmax=10, num_atoms=51):
    """
    Categorical value functions.
    Adds new dimension to representation tensor with num_atoms elements.
    Each output corresponds to probability to receive reward from categorical distribution
    with grid support from -10 to 10 with num_atoms cells.
    Based on: https://arxiv.org/pdf/1707.06887.pdf
    
    Args:
        Vmin - minimum value of approximation distribution, int
        Vmax - maximum value of approximation distribution, int
        num_atoms - number of atoms in approximation distribution, int
    """
    assert Vmin < Vmax, "Vmin must be less than Vmax!"
    assert issubclass(parclass, V)
    
    support = torch.linspace(Vmin, Vmax, num_atoms).refine_names("atoms")
    delta_z = float(Vmax - Vmin) / (num_atoms - 1)

    class CategoricalValue(parclass):
        def expectation(self):
            '''
            Reduces "atoms" dimension by computing expectation
            output: V (atoms dimension reduced)
            '''
            probabilities = F.softmax(self.tensor, dim="atoms")
            outcomes = support.to(self.tensor.device).align_as(self.tensor)
            return self.construct((probabilities * outcomes).sum(dim="atoms"))

        def one_step(self, rewards, discounts):
            '''
            Performs some magic concerning projecting shifted and squeezed categorical
            distribution back to the grid (Vmin ... Vmax) with num_atoms.
            input: Reward
            input: Discount
            output: V (dimensions not changed)
            '''
            distributions = F.softmax(self.tensor, dim="atoms").align_to(..., "atoms")
            rewards = rewards.tensor.align_as(distributions).rename(None)
            discounts = discounts.tensor.align_as(distributions).rename(None)                
            supports = support.to(self.tensor.device).align_as(distributions).rename(None)

            names = distributions.names
            distributions = distributions.rename(None)
            
            Tz = rewards + discounts * supports
            Tz = Tz.clamp(min=Vmin, max=Vmax)
            b  = (Tz - Vmin) / delta_z
            l  = b.floor().long()
            u  = b.ceil().long()

            numel = self.tensor.numel() // num_atoms
            offset = torch.linspace(0, (numel - 1) * num_atoms, numel).long().view(-1, 1).to(self.tensor.device)
            
            proj_dist = torch.zeros_like(distributions)
            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (distributions * (u.float() + (b.ceil() == b).float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (distributions * (b - l.float())).view(-1))
            proj_dist /= proj_dist.sum(-1, keepdims=True)

            proj_dist = proj_dist.log()                
            proj_dist = proj_dist.refine_names(*names).align_as(self.tensor)
            return self.construct(proj_dist)

        def compare(self, target):
            '''
            Calculates KL-divergence between target and this Categorical value.
            input: target - same dimensions
            output: Loss
            '''
            target_distribution = F.softmax(target.tensor, dim="atoms")
            distribution = torch.clamp(F.softmax(self.tensor, dim="atoms"), 1e-8, 1 - 1e-8)
            loss = -(target_distribution * distribution.log()).sum("atoms")
            return self.mdp["Loss"](loss)

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
            dims["atoms"] = lambda parclass: Categorical(parclass, Vmin, Vmax, num_atoms)
            return dims

        @classmethod
        def _default_name(cls):    
            return super()._default_name() + ' in categorical from with {} atoms from {} to {}'.format(num_atoms, Vmin, Vmax)
    return CategoricalValue