from .qHead import *

def CategoricalQ(Vmin=-10, Vmax=10, num_atoms=51):
    """
    Categorical DQN.
    Based on: https://arxiv.org/pdf/1707.06887.pdf
    
    Args:
        Vmin - minimum value of approximation distribution, int
        Vmax - maximum value of approximation distribution, int
        num_atoms - number of atoms in approximation distribution, int
    """
    assert Vmin < Vmax, "Vmin must be less than Vmax!"
    support = torch.linspace(Vmin, Vmax, num_atoms).to(device)

    def CategoricalQ(system): 
        class CategoricalQ(Q(system)):
            '''
            FloatTensor representing categorical Q-function, (*batch_shape x num_actions x num_atoms)
            where support is a grid (i.e. from -10 to 10 with num_atoms cells)
            and each output corresponds to probability to receive reward from this cell
            if selecting action a in state s
            '''
            def __init__(self, x):
                assert x.shape[-1] == type(self).required_shape()
                x = x.view(*x.shape[:-1], system.num_actions, num_atoms)
                self.tensor = F.softmax(x, dim=-1)                

            def required_shape():
                return system.num_actions * num_atoms

            def greedy(self):
                '''
                output: LongTensor, (*batch_shape x num_atoms)
                '''
                return (self.tensor * support).sum(-1).max(-1)[1]
                
            def gather(self, action_b):
                '''
                input: action_b - LongTensor, (*batch_shape)
                output: FloatTensor, (*batch_shape)
                '''
                return self.tensor.gather(-2, action_b[..., None, None].expand(*self.tensor.shape[:-2], 1, num_atoms)).squeeze(-2)
            
            def value(self):
                '''
                output: FloatTensor, (*batch_shape x num_atoms)
                '''
                return self.gather(self.greedy())

            def one_step_q(batch, next_v):
                '''
                Performs some magic concerning projecting shifted and squeezed categorical distribution
                back to the grid (Vmin ... Vmax) with num_atoms.
                input: Batch
                input: next_v, (*batch_shape, num_atoms)
                output: FloatTensor, (*batch_shape, num_atoms)
                '''

                # TODO does it work with several-dimensional batches?
                # taken from source code
                offset = torch.linspace(0, (len(batch) - 1) * num_atoms, len(batch)).long().unsqueeze(1).expand(len(batch), num_atoms).to(device)
                delta_z = float(Vmax - Vmin) / (num_atoms - 1)
                
                reward_b = batch.reward.unsqueeze(1).expand_as(next_v)
                done_b   = batch.done.unsqueeze(1).expand_as(next_v)
                support_grid = support.unsqueeze(0).expand_as(next_v)

                Tz = reward_b + (1 - done_b) * (system.gamma**batch.n_steps) * support_grid
                Tz = Tz.clamp(min=Vmin, max=Vmax)
                b  = (Tz - Vmin) / delta_z
                l  = b.floor().long()
                u  = b.ceil().long()        
                
                proj_dist = Tensor(next_v.size()).zero_()              
                proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_v * (u.float()+ (b.ceil() == b).float() - b)).view(-1))
                proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_v * (b - l.float())).view(-1))
                proj_dist /= proj_dist.sum(1).unsqueeze(1)
                return proj_dist

            def compare(q, guess):
                '''
                Calculates batch loss
                input: guess - target, FloatTensor, (*batch_shape, num_atoms)
                input: q - current model output, FloatTensor, (*batch_shape, num_atoms)
                output: FloatTensor, (*batch_shape)
                '''
                q.data.clamp_(1e-8, 1 - 1e-8)   # TODO doesn't torch have cross entropy? Taken from source code.
                return -(guess * q.log()).sum(-1)

            def __repr__(self):    
                return 'Categorical Q-function for {} actions with {} atoms from {} to {}'.format(system.num_actions, num_atoms, Vmin, Vmax)
        return CategoricalQ
    return CategoricalQ