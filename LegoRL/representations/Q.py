from LegoRL.representations.V import V
from LegoRL.representations.standard import Action

import torch
from LegoRL.utils.namedTensorsUtils import torch_gather

def StateActionV(parclass):
    class Quality(parclass):
        """
        State-action value function (Quality function, Q-function) representation.

        Adds action dimension to representation tensor.
        """
        def greedy(self):
            '''
            Returns greedy action.
            output: Action
            '''
            return self.mdp[Action](self.tensor.max(dim="actions").indices)
            
        def gather(self, actions):
            '''
            Selectes this representation for given actions.
            input: actions - Action, (*batch_shape)
            output: V (actions dimension reduced)
            '''
            return self.construct(torch_gather(self.tensor, actions.tensor, "actions"))
        
        def value(self, policy=None):
            '''
            Returns optimal value if policy is None, expectation otherwise.
            input: policy - None or Policy
            output: V (actions dimension reduced)
            '''
            if policy is None:
                return self.construct(self.tensor.max(dim="actions").values)
            return self.construct((self.tensor * policy.proba.align_as(self.tensor)).sum("actions"))

        def scalar(self):
            return self.value().scalar()
        
        @classmethod
        def rshape(cls):
            return torch.Size([cls.mdp.num_actions]) + super().rshape()
        
        @classmethod
        def rnames(cls):
            return ("actions",) + super().rnames()

        @classmethod
        def constructor(cls):
            dims = super().constructor()
            dims["actions"] = StateActionV
            return dims

        def __repr__(self):    
            return f'Q-function for {self.mdp.num_actions} actions'
    return Quality

# convinient reduction
Q = StateActionV(V)