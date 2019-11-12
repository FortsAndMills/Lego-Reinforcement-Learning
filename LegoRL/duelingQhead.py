from .qHead import *
from .categoricalQhead import *

def DuelingQ(system):
    """
    Dueling DQN head.
    Based on: https://arxiv.org/abs/1511.06581
    """
    class DuelingQ(Q(system)):
        '''
        FloatTensor representing Q-function using dueling head:
            with A-head, (*batch_shape x num_actions) and
                 V-head, (*batch_shape x 1)
            which are aggregated using heuristic
            Q = V + A - A.mean(dim=-1)
        '''
        def __init__(self, x):
            assert x.shape[-1] == type(self).required_shape()

            v, a = torch.split(x, [1, system.num_actions], dim=-1)
            self.tensor = v + a - a.mean(dim=-1, keepdim=True)

        def required_shape():
            return system.num_actions + 1

        def __repr__(self):    
            return 'Q-function in dueling form (V + A - A.mean()) for {} actions'.format(system.num_actions)
    return DuelingQ

def DuelingCategoricalQ(Vmin=-10, Vmax=10, num_atoms=51):
    def DuelingCategoricalQ(system):
        """
        Dueling DQN head as proposed in Rainbow DQN.
        Based on: https://arxiv.org/abs/1710.02298
        """
        class DuelingCategoricalQ(CategoricalQ(Vmin, Vmax, num_atoms)(system)):
            '''
            FloatTensor representing categorical Q-function using dueling head:
                with A-head, (*batch_shape x num_actions x num_atoms) and
                    V-head, (*batch_shape x 1 x num_atoms)
                which are aggregated using heuristic
                Q = softmax(V + A - A.mean(dim=action_dim), dim=atoms_dim)
            '''
            def __init__(self, x):
                assert x.shape[-1] == type(self).required_shape()

                x = x.view(*x.shape[:-1], system.num_actions + 1, num_atoms)
                v, a = torch.split(x, [1, system.num_actions], dim=-2)
                q = v + a - a.mean(dim=-2, keepdim=True)
                self.tensor = F.softmax(q, dim=-1)

            def required_shape():
                return (system.num_actions + 1) * num_atoms

            def __repr__(self):
                return 'Categorical Q-function in dueling form (V + A - A.mean()) for {} actions with {} atoms from {} to {}'.format(system.num_actions, num_atoms, Vmin, Vmax)
        return DuelingCategoricalQ
    return DuelingCategoricalQ