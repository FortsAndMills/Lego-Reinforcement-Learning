import torch
import torch.nn.functional as F

def Categorical(Vmin=-10, Vmax=10, num_atoms=51):
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
    
    support = torch.linspace(Vmin, Vmax, num_atoms)#.to(device)
    support = support.refine_names("atoms")
    delta_z = float(Vmax - Vmin) / (num_atoms - 1)

    def Categorical(parclass):
        class CategoricalValue(parclass):
            @classmethod
            def shape(cls, system):
                return torch.Size((num_atoms,)) + super().shape(system)
        
            @classmethod
            def names(cls):
                return ("atoms",) + super().names()

            @classmethod
            def constructor(cls):
                dims = super().constructor()
                dims["atoms"] = Categorical
                return dims

            def _expectation(self):
                '''
                Reduces atoms dimension.
                output: V without atoms dimension
                '''
                probabilities = F.softmax(self.tensor, dim="atoms")
                outcomes = support.to(self.system.device).align_as(self.tensor)
                return self.construct((probabilities * outcomes).sum(dim="atoms"))

            def one_step(self, batch):
                '''
                Performs some magic concerning projecting shifted and squeezed categorical
                distribution back to the grid (Vmin ... Vmax) with num_atoms.
                input: Batch
                output: V (dimensions not changed)
                '''
                distributions = F.softmax(self.tensor, dim="atoms").align_to(..., "atoms")
                rewards = batch.rewards.align_as(distributions).rename(None)
                discounts = batch.discounts.align_as(distributions).rename(None)                
                supports = support.to(self.system.device).align_as(distributions).rename(None)

                names = distributions.names
                distributions = distributions.rename(None)
                
                Tz = rewards + discounts * supports
                Tz = Tz.clamp(min=Vmin, max=Vmax)
                b  = (Tz - Vmin) / delta_z
                l  = b.floor().long()
                u  = b.ceil().long()

                numel = self.tensor.numel() // num_atoms
                offset = torch.linspace(0, (numel - 1) * num_atoms, numel).long().view(-1, 1).to(self.system.device)
                
                proj_dist = torch.zeros_like(distributions)
                proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (distributions * (u.float() + (b.ceil() == b).float() - b)).view(-1))
                proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (distributions * (b - l.float())).view(-1))
                proj_dist /= proj_dist.sum(-1, keepdims=True)
                
                proj_dist = proj_dist.refine_names(*names).align_as(self.tensor)
                return self.construct(proj_dist)

            def compare(self, target):
                '''
                Calculates KL-divergence between target and this Categorical value.
                input: target - CategoricalV
                output: FloatTensor, (*batch_shape)
                '''
                # TODO doesn't torch have cross entropy? Taken from source code.
                target_distribution = F.softmax(target.tensor, dim="atoms")
                distribution = torch.clamp(F.softmax(self.tensor, dim="atoms"), 1e-8, 1 - 1e-8)
                return -(target_distribution * distribution.log()).sum("atoms")

            def greedy(self):
                return self._expectation().greedy()
                
            def value(self, policy=None):
                if policy is None:
                    return self.gather(self.greedy())
                return super().value(policy)

            def __repr__(self):    
                return super().__repr__() + ' in categorical from with {} atoms from {} to {}'.format(num_atoms, Vmin, Vmax)
        return CategoricalValue
    return Categorical